FROM ubuntu:24.04

RUN apt update && apt install -y \
    build-essential \
    cmake \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-program-options-dev \
    libboost-test-dev \
    libeigen3-dev \
    zlib1g-dev \
    libbz2-dev \
    lzma-dev \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=True

ENV APP_HOME=/app
WORKDIR $APP_HOME

# Install kenlm
RUN git clone https://github.com/kpu/kenlm.git
WORKDIR $APP_HOME/kenlm
RUN mkdir build
WORKDIR $APP_HOME/kenlm/build
RUN cmake ..
RUN make -j 4

# Download models
WORKDIR $APP_HOME
RUN wget -c -P models http://dl.fbaipublicfiles.com/cc_net/lm/ja.arpa.bin
RUN wget -c -P models http://dl.fbaipublicfiles.com/cc_net/lm/ja.sp.model

# Install python packages
COPY ./requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY ./ng_word.txt ./ng_word.txt
COPY ./pre_filter.py ./pre_filter.py
COPY ./data ./data
COPY ./output ./output

ENTRYPOINT ["python3", "./pre_filter.py"]
