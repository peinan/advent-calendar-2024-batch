import json
import os
import sys
import gc
from typing import Any
from tqdm import tqdm
import unicodedata
import psutil
import argparse

from hojichar import Compose, document_filters, deduplication, Parallel, Document
from hojichar.filters.document_filters import JSONLoader
from hojichar.core.filter_interface import Filter

from huggingface_hub import hf_hub_download
import time


class OscarDocument(Document):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = {}


class SpaceFilter(Filter):
    def apply(self, doc):
        space_count = 20
        text = doc.text
        if (len(text) > 100):
            ## 半角スペース or 全角スペースを多く含む
            if (text.count(' ') > space_count or text.count('　') > space_count):
                doc.is_rejected = True

        doc.text = text
        return doc


class FilterByQualityWarnings(Filter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality_key = 'quality_warnings'

    def apply(self, doc: OscarDocument):
        if not self.quality_key in doc.metadata:
            return doc
        quality = doc.metadata[self.quality_key]
        if quality is None:
            return doc
        if 'header' in quality or 'footer' in quality or 'noisy' in quality:
            doc.is_rejected = True

        return doc


class PPLFilter(Filter):
    def __init__(self, model_path, sp_model_path, ppl_th, *args: Any, **kwargs: Any) -> None:
        import kenlm
        import sentencepiece

        super().__init__(*args, **kwargs)
        self.ppl_th = ppl_th
        self.model = kenlm.LanguageModel(model_path)
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(sp_model_path)

    def apply(self, document):
        text = document.text
        text = unicodedata.normalize('NFD', text)
        toks = self.sp.encode(text, out_type=str)

        sentence = " ".join(toks)
        ppl = self.model.perplexity(sentence)
        if ppl > self.ppl_th:
            # print(ppl, document.text)
            document.is_rejected = True
        return document


class OscarJSONLoader(JSONLoader):
    def __init__(self, metadata_keys=[], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.meta = 'metadata'
        self.metadata_keys = metadata_keys

    def apply(self, document):
        try:
            data = json.loads(document.text)
            document.text = str(data[self.key])
            for k in self.metadata_keys:
                document.metadata[k] = data[self.meta][k]

        except Exception as e:
            if self.ignore:
                document.is_rejected = True
                return document
            else:
                raise e

        return document


class Debug(Filter):
    def __init__(self, idx="", *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.idx = idx

    def apply(self, document):
        print(self.idx)
        print(document.text)
        print(document.is_rejected)
        print('**' * 40)
        return document


class Timer(Filter):
    def __init__(self, start, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.start = start

    def apply(self, document):
        print('time: ', time.time() - self.start)
        return document


def extract_zst_file(input_file, output_file):
    import zstandard as zstd
    with open(input_file, 'rb') as compressed_file:
        decompressor = zstd.ZstdDecompressor()
        with decompressor.stream_reader(compressed_file) as reader, open(output_file, 'wb') as output:
            while True:
                chunk = reader.read(16384)
                if not chunk:
                    break
                output.write(chunk)
                del chunk
        del reader
        del output
    del compressed_file
    gc.collect()


def read_yielder(input_file):
    cnt = 0
    with open(input_file) as fp:
        for line in fp.readlines():
            # if cnt > 10000:
            #     break
            cnt += 1
            yield OscarDocument(line)


def show_diff_mem(num, start):
    def format(size):
        power = 2 ** 10
        n = 0
        power_labels = {0: '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
        while size > power:
            size /= power
            n += 1
        return size, power_labels[n] + 'bytes'
        # print(num, format(psutil.virtual_memory().used - start))

    print(num, format(psutil.virtual_memory().used))


def clean(input_file, output_file, num_jobs=10):
    key = 'text'
    key = 'content'
    # before_debup_file = './data/before_debup.jsonl'
    before_debup_file = output_file
    start = psutil.virtual_memory().used

    cleaner = Compose([
        OscarJSONLoader(key=key, metadata_keys=['quality_warnings']),
        document_filters.DocumentLengthFilter(min_doc_len=100, max_doc_len=50000),
        document_filters.AcceptJapanese(),
        FilterByQualityWarnings(),
        SpaceFilter(),
        document_filters.NgWordsFilterJa(dict_path='./ng_word.txt'),
        document_filters.DiscardBBSComments(),
        document_filters.DiscardAds(),
        document_filters.DocumentNormalizer(),
        document_filters.MaskPersonalInformation(),
        PPLFilter(
            model_path=kenlm_model_path,
            sp_model_path=sp_model_path,
            ppl_th=90000
        ),
        document_filters.JSONDumper()
    ])

    print('-- start clean --')
    cnt = 0

    total_lines = 0
    show_diff_mem(0.5, start)
    with open(input_file, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for _ in file)
    gc.collect()
    show_diff_mem(1, start)

    print('raw data len ', total_lines)
    t = tqdm(total=total_lines)
    with Parallel(cleaner, num_jobs=num_jobs) as pfilter:
        show_diff_mem(2, start)
        with open(before_debup_file, "w") as fp:
            for doc in pfilter.imap_apply(read_yielder(input_file)):
                if not doc.is_rejected:
                    fp.write(doc.text + "\n")
                    cnt += 1
                del doc
                t.update(1)
        t.close()
    show_diff_mem(4, start)
    print('end data len: ', cnt)
    gc.collect()
    show_diff_mem(5, start)


def main():
    print('start...')
    print(f'start: {start_part}')
    print(f'end: {end_part}')
    print(f'num_jobs: {num_jobs}')

    for i in range(start_part, end_part + 1):
        url = f'https://huggingface.co/datasets/oscar-corpus/OSCAR-2301/resolve/main/ja_meta/ja_meta_part_{i}.jsonl.zst'
        print('get...', url)
        zst_file_name = os.path.basename(url)
        hf_hub_download(repo_id='oscar-corpus/OSCAR-2301',
                        subfolder='ja_meta',
                        local_dir=input_dir,
                        filename=zst_file_name,
                        repo_type="dataset",
                        token=args.hf_token
                        )
        input_ex_file = input_dir + '/ja_meta/' + zst_file_name
        jsonl_file = os.path.splitext(input_ex_file)[0]
        show_diff_mem(0, start_part)
        extract_zst_file(input_ex_file, jsonl_file)
        show_diff_mem(0.1, start_part)
        output_file = f'{output_dir}/{i}.jsonl'

        print('input...', jsonl_file)
        print('output...', output_file)
        clean(jsonl_file, output_file, num_jobs=num_jobs)
        gc.collect()
        show_diff_mem(8, start_part)


def test():
    clean('./data/sample_input.jsonl', './output/sample_output.jsonl')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=119)
    parser.add_argument('--hf-token', type=str, default='')
    parser.add_argument('--input', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--ng-word', type=str, default='./ng_word.txt')
    parser.add_argument('--kenlm-model', type=str, default='./models/ja.arpa.bin')
    parser.add_argument('--sentencepiece-model', type=str, default='./models/ja.sp.model')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--sample-run', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)

    start_part = args.start
    end_part = args.end
    hf_token = args.hf_token
    input_dir = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    ng_word_filepath = args.ng_word
    kenlm_model_path = args.kenlm_model
    sp_model_path = args.sentencepiece_model
    num_jobs = args.workers

    if args.sample_run:
        test()
    else:
        if hf_token == '':
            print('Please set the Hugging Face token.')
            sys.exit(1)

        main()
