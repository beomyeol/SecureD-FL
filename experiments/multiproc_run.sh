#!/bin/bash

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && pwd)"

. ${SRC_DIR}/setup.sh

PROG_NAME="decentralized_simulation_main.py"

# LeNet
MODEL="lenet"
DATASET="femnist"
BATCH_SIZE=32

LOG_EVERY_N_STEPS=50

# fedavg
EPOCHS=10
LOCAL_EPOCHS=10
LOG_DIR="$(get_log_dir)/fedavg"
run_with_ckpt "$(get_cmd)" "${LOG_DIR}"
