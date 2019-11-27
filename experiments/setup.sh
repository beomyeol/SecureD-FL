#!/bin/bash

PYTHON="python3"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
EXPERIMENT_DIR="${ROOT_DIR}/../experiments"
DATASET_ROOT_DIR="${ROOT_DIR}/../data"

MODEL=""
DATASET=""

NUM_WORKERS=9

EPOCHS=1
LOCAL_EPOCHS=1
LOG_EVERY_N_STEPS=1

BATCH_SIZE=1

function get_dataset_dir() {
  echo "${DATASET_ROOT_DIR}/$1"
}

function get_cmd() {
  local CMD="$PYTHON $ROOT_DIR/sequential_simulation_main.py \
    --dataset_dir=$(get_dataset_dir ${DATASET}) \
    --dataset_download \
    --model=$MODEL \
    --epochs=$EPOCHS \
    --local_epochs=$LOCAL_EPOCHS \
    --log_every_n_steps=$LOG_EVERY_N_STEPS \
    --num_workers=$NUM_WORKERS \
    --batch_size=$BATCH_SIZE \
    --validation_period=1 \
    --weighted_avg"
  echo $CMD
}

function run() {
  local CMD=$1
  local LOG_PATH=$2

  local LOG_PATH_DIR=$(dirname $LOG_PATH)

  if ! [ -d $LOG_PATH_DIR ]; then
    mkdir -p $LOG_PATH_DIR
  fi

  echo $CMD
  echo $CMD >$LOG_PATH
  $CMD |& tee -a $LOG_PATH
}

function get_log_dir() {
  echo "$EXPERIMENT_DIR/$MODEL/${NUM_WORKERS}_workers/bs_${BATCH_SIZE}/${EPOCHS}_${LOCAL_EPOCHS}"
}

function get_admm_args() {
  echo "--use_admm \
    --secure_admm \
    --admm_threshold=$ADMM_THRESHOLD \
    --admm_lr=$ADMM_LR \
    --admm_decay_rate=$ADMM_DECAY_RATE \
    --admm_decay_period=$ADMM_DECAY_PERIOD \
    --admm_max_iter=$ADMM_MAX_ITER"
}

function run_with_ckpt() {
  local CMD=$1
  local LOG_DIR=$2

  run "${CMD} --save_dir=${LOG_DIR} --save_period=1" "${LOG_DIR}/run.log"
}

function run_with_ckpt_gpus() {
  local CMD=$1
  local LOG_DIR=$2
  local NUM_GPUS=$3
  for ((i = 0; i < $NUM_GPUS; i++)); do
    run_with_ckpt "${CMD} --gpu_id=$i --seed=$RANDOM" "${LOG_DIR}/$i" &
  done
  wait
}
