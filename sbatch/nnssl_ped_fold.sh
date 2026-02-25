#!/bin/bash
#SBATCH --job-name=nnssl-ped
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --output=/gpfs/share/home/2401111663/syy/nnssl-openneuro/sbatch/output/ped/nnssl-ped-%j.out
#SBATCH --error=/gpfs/share/home/2401111663/syy/nnssl-openneuro/sbatch/error/ped/nnssl-ped-%j.err

set -euo pipefail

# Find Python binary
PYTHON_BIN=/gpfs/share/home/2401111663/anaconda3/envs/syy1/bin/python

# Directory setup
NNSSL_ROOT=/gpfs/share/home/2401111663/syy/nnssl-openneuro
DATA_ROOT=/gpfs/share/home/2401111663/syy/braTS_5folds/ped_task/5_fold
export PYTHONPATH=${NNSSL_ROOT}/src:${PYTHONPATH:-}

# PLEASE SET YOUR PRETRAINED WEIGHTS HERE
PRETRAINED=${NNSSL_ROOT}/checkpoint_final_mae.pth

for FOLD in {1..5}; do
  FOLD_DIR=$DATA_ROOT/fold_${FOLD}
  JSON_OUT=$DATA_ROOT/dataset_ped_fold_${FOLD}_mae.json
  OUTPUT_DIR="${NNSSL_ROOT}/training_runs/ped_fold${FOLD}_mae"
  mkdir -p ${OUTPUT_DIR}

  echo "Starting PED fold ${FOLD} with MAE (ResEncL)"

  # Prepare JSON format
  $PYTHON_BIN ${NNSSL_ROOT}/src/nnssl/utilities/prepare_dataset_json.py \
      --train-json $FOLD_DIR/train.json \
      --val-json $FOLD_DIR/val.json \
      --output-json $JSON_OUT

  $PYTHON_BIN ${NNSSL_ROOT}/src/nnssl/evaluation/segmentation3d_nnssl.py \
      --dataset-name BraTS \
      --datalist-path $JSON_OUT \
      --data-root / \
      --pretrained-weights $PRETRAINED \
      --arch ResEncL \
      --epochs 200 \
      --epoch-length 24 \
      --batch-size 2 \
      --learning-rate 0.0001 \
      --output-dir $OUTPUT_DIR
done
