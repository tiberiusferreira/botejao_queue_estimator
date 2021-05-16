#!/bin/zsh
set -e
export LIBTORCH=/Users/tiberio/Documents/github/botejao_queue_estimator/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
cargo build
