#!/bin/bash

. setup.sh

ROOT_DIR="${HOME}/federated"
EXPERIMENT_DIR="${HOME}/experiments"
DATA_ROOT_DIR="${HOME}/data"

NUM_GPUS=4

# LeNet
MODEL="lenet"
DATASET_DIR="$DATA_ROOT_DIR/femnist"
BATCH_SIZE=32

# RNN 
#MODEL="rnn"
#BATCH_SIZE=1
#DATASET_DIR="$DATA_ROOT_DIR/shakespeare"

##############################################

# fedavg
EPOCHS=10
LOCAL_EPOCHS=10
LOG_DIR="$(get_log_dir)/fedavg"
run_with_ckpt_gpus "$(get_cmd)" "${LOG_DIR}" "${NUM_GPUS}"

# local only
EPOCHS=1
LOCAL_EPOCHS=100
LOG_DIR="$(get_log_dir)/local_only"
#run_with_ckpt_gpus "$(get_cmd)" "${LOG_DIR}" "${NUM_GPUS}"

# ADMM
EPOCHS=10
LOCAL_EPOCHS=10
ADMM_MAX_ITER=10
ADMM_LR=0.005
ADMM_DECAY_RATE=1
ADMM_DECAY_PERIOD=12
ADMM_THRESHOLD=1e-6
LOG_DIR="$(get_log_dir)/admm_${ADMM_LR}_${ADMM_DECAY_RATE}_${ADMM_DECAY_PERIOD}_${ADMM_THRESHOLD}"
#run_with_ckpt_gpus "$(get_cmd) $(get_admm_args)" "${LOG_DIR}" "${NUM_GPUS}"

