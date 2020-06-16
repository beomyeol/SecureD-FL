#!/bin/bash

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

. ${SRC_DIR}/setup.sh

# LeNet
MODEL="lenet"
DATASET="femnist"
BATCH_SIZE=32

# RNN
# MODEL="rnn"
# BATCH_SIZE=1
# DATASET="shakespeare"

# CifarNet
# MODEL="cifarnet"
# DATASET="cifar10"
# BATCH_SIZE=32

# ResNet18
# MODEL="resnet18"
# DATASET="cifar10"
# BATCH_SIZE=32

LOG_EVERY_N_STEPS=50

##############################################

# fedavg
EPOCHS=10
LOCAL_EPOCHS=10
LOG_DIR="$(get_log_dir)/fedavg"
run_with_ckpt "$(get_cmd)" "${LOG_DIR}"

# local only
EPOCHS=1
LOCAL_EPOCHS=100
LOG_DIR="$(get_log_dir)/local_only"
run_with_ckpt "$(get_cmd)" "${LOG_DIR}"

# ADMM (7)
EPOCHS=10
LOCAL_EPOCHS=10
ADMM_MAX_ITER=7
ADMM_LR=0.1
ADMM_DECAY_RATE=0.9
ADMM_DECAY_PERIOD=4
ADMM_THRESHOLD=0
LOG_DIR="$(get_log_dir)/admm_${ADMM_MAX_ITER}_${ADMM_LR}_${ADMM_DECAY_RATE}_${ADMM_DECAY_PERIOD}_${ADMM_THRESHOLD}"
run_with_ckpt "$(get_cmd) $(get_admm_args) --secure_admm" "${LOG_DIR}"

# ADMM (4)
EPOCHS=10
LOCAL_EPOCHS=10
ADMM_MAX_ITER=4
ADMM_LR=0.005
ADMM_DECAY_RATE=0.9
ADMM_DECAY_PERIOD=2
ADMM_THRESHOLD=0
LOG_DIR="$(get_log_dir)/admm_${ADMM_MAX_ITER}_${ADMM_LR}_${ADMM_DECAY_RATE}_${ADMM_DECAY_PERIOD}_${ADMM_THRESHOLD}"
run_with_ckpt "$(get_cmd) $(get_admm_args) --secure_admm" "${LOG_DIR}"
