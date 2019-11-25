FROM rust:1.39


RUN mkdir darknet
WORKDIR ./darknet-rs
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.3.1%2Bcpu.zip
RUN apt update && apt install -y unzip
RUN unzip libtorch-cxx11-abi-shared-with-deps-1.3.1+cpu.zip
RUN rustup default nightly

COPY ./src ./src
COPY ./Cargo.toml .
COPY ./yolo-v3.ot .
COPY ./yolo-v3.cfg .

RUN env LIBTORCH=$PWD/libtorch LD_LIBRARY_PATH=$PWD/libtorch/lib:$LD_LIBRARY_PATH cargo build

CMD ["env LIBTORCH=$PWD/libtorch LD_LIBRARY_PATH=$PWD/libtorch/lib:$LD_LIBRARY_PATH cargo run"]

