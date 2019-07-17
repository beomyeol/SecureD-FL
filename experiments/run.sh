#!/bin/bash

HOME_DIR="/home/beomyeol"
ROOT_DIR="$HOME_DIR/federated"
EXPERIMENT_DIR="$HOME_DIR/experiments"
DATA_ROOT_DIR="$HOME_DIR/data"
PYTHON="python3"

MODEL="lenet"
DATASET_DIR="$DATA_ROOT_DIR/femnist"
EPOCHS=10
LOCAL_EPOCHS=10
LOG_EVERY_N_STEPS=500
BATCH_SIZE=64

function get_cmd {
  local CMD="$PYTHON $ROOT_DIR/sequential_simulation_main.py \
    --dataset_dir=$DATASET_DIR
    --model=$MODEL
    --epochs=$EPOCHS
    --local_epochs=$LOCAL_EPOCHS
    --log_every_n_etsps=$LOG_EVERY_N_STEPS
    --validation_period=1"
  echo $CMD
}

function run {
  local CMD=$1
  local LOG_PATH=$2

  echo $CMD > $LOG_PATH
  $CMD |& tee -a $LOG_PATH
}
