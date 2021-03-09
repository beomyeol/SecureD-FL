#!/bin/bash

ROOT_DIR="../earhart/experiments"

NUM_WORKERS=9
# NUM_WORKERS=15

BATCH_SIZE=32
MODEL="lenet"
# MODEL="rnn"
# MODEL="resnet18"
OPTIMIZER="rmsprop"
# OPTIMIZER="adam"
LR=0.001
GLOBAL_EPOCHS=50
LOCAL_EPOCHS=1
YTICKS="0.95,1,0.01"
# YTICKS="0.3,0.55,0.05"
# YTICKS="0.2,0.8,0.1"

BASE_DIR="${ROOT_DIR}/${MODEL}/${NUM_WORKERS}_workers/bs_${BATCH_SIZE}/${OPTIMIZER}/lr_${LR}/${GLOBAL_EPOCHS}_${LOCAL_EPOCHS}"

FEDAVG_LOG_DIR="${BASE_DIR}/fedavg/logs"
# ADMM_LOG_DIR="${BASE_DIR}/admm_4_0.005_0.9_2_0/logs"
ADMM_LOG_DIR="${BASE_DIR}/admm_2_3_0.0003_1_0/logs"
LOCALONLY_LOG_DIR="${BASE_DIR}/local_only/logs"

OUT_DIR="./plots/${MODEL}/${NUM_WORKERS}_workers/bs_${BATCH_SIZE}/${OPTIMIZER}/lr_${LR}/${GLOBAL_EPOCHS}_${LOCAL_EPOCHS}"
mkdir -p ${OUT_DIR}

for TAG in "validation_accuracy (after aggr.)" #"train_loss" "train_accuracy"
do
  if [ "${TAG}" = "validation_accuracy (after aggr.)" ]; then
    OUTPUT_TAG="validation_accuracy_after_aggr"
    ADDITIONAL_ARGS="--yticks ${YTICKS}"
  else
    OUTPUT_TAG=${TAG}
    ADDITIONAL_ARGS=""
  fi
  python ./scripts/plot_summary.py \
    --label Local-Only FedAvg SecureD-FL \
    --logdir ${LOCALONLY_LOG_DIR} ${FEDAVG_LOG_DIR} ${ADMM_LOG_DIR}\
    --name "${TAG}" \
    --output ${OUT_DIR}/${OUTPUT_TAG}.png \
    ${ADDITIONAL_ARGS}
done

# for TAG in "validation_accuracy (after aggr.)"
# do
  # if [ "${TAG}" = "validation_accuracy (after aggr.)" ]; then
    # OUTPUT_TAG="validation_accuracy_after_aggr"
    # ADDITIONAL_ARGS="--yticks ${YTICKS}"
  # else
    # OUTPUT_TAG=${TAG}
    # ADDITIONAL_ARGS=""
  # fi
  # python ./scripts/plot_summary.py \
    # --label SecureD-FL\(2\) SecureD-FL\(3\) SecureD-FL\(4\) SecureD-FL\(5\) SecureD-FL\(6\) SecureD-FL\(7\)\
    # --logdir \
      # ${BASE_DIR}/admm_2_3_0.0003_1_0/logs \
      # ${BASE_DIR}/admm_3_7_0.01_1_0/logs \
      # ${BASE_DIR}/admm_4_7_0.0003_2_0/logs \
      # ${BASE_DIR}/admm_5_50_7e-6_3_0/logs \
      # ${BASE_DIR}/admm_6_100_7e-6_3_0/logs \
      # ${BASE_DIR}/admm_7_100_7e-6_4_0/logs \
    # --name "${TAG}" \
    # --output ${OUT_DIR}/admm_${OUTPUT_TAG}.png \
    # ${ADDITIONAL_ARGS}
# done


