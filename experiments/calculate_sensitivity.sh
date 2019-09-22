#!/bin/bash

ROOT_DIR=""

DATASET_NAME="femnist"

DATASET_DIR=""

BATCH_SIZE=1024
NUM_PROCS=4
NUM_GPUS=1

CMD_BASE="python ${ROOT_DIR}/calculate_sensitivity.py \
  --name=${DATASET_NAME} \
  --dataset_dir=${DATASET_DIR} \
  --batch_size=${BATCH_SIZE} \
  --num=${NUM_PROCS}"

for ((i=0;i<$NUM_PROCS;i++))
do
  GPU_ID=`expr $i % ${NUM_GPUS}`
  CMD="$CMD_BASE --gpu_id=${GPU_ID} --id=$i"
  $CMD &
done
wait
