#!/bin/bash
#SBATCH --job-name=nnssl-v3-ped
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --output=/gpfs/share/home/2401111663/syy/nnssl-openneuro/sbatch/output/ped/nnssl-v3-%j.out
#SBATCH --error=/gpfs/share/home/2401111663/syy/nnssl-openneuro/sbatch/error/ped/nnssl-v3-%j.err

set -euo pipefail

# Find Python binary
PYTHON_BIN=/gpfs/share/home/2401111663/anaconda3/envs/syy1/bin/python

# Directory setup
NNSSL_ROOT=/gpfs/share/home/2401111663/syy/nnssl-openneuro
BRAIN_MVP_JSON_DIR=/gpfs/share/home/2401111663/syy/braTS_5folds/pediatric_task/5_fold
export PYTHONPATH=${NNSSL_ROOT}/src:${PYTHONPATH:-}

# PRETRAINED WEIGHTS
PRETRAINED=${NNSSL_ROOT}/checkpoint_final_mae.pth

# Folders to iterate
FOR_FOLDS=(2 3 4 5)

for FOLD in "${FOR_FOLDS[@]}"; do
  echo "========== Starting NNSSL Fold ${FOLD} =========="
  
  # Input JSONs from the v3 generation (Real and Mixed)
  JSON_V3_REAL=$BRAIN_MVP_JSON_DIR/fold_${FOLD}/ped_real_fold${FOLD}_v3.json
  JSON_V3_MIXED=$BRAIN_MVP_JSON_DIR/fold_${FOLD}/ped_mixed_fold${FOLD}_v3.json
  
  # Original val.json for test set
  JSON_ORIG_VAL=$BRAIN_MVP_JSON_DIR/fold_${FOLD}/val.json
  
  # Target JSONs for NNSSL logic
  NNSSL_JSON_REAL=$NNSSL_ROOT/jsons/ped_real_fold${FOLD}_v3_nnssl.json
  NNSSL_JSON_MIXED=$NNSSL_ROOT/jsons/ped_mixed_fold${FOLD}_v3_nnssl.json
  mkdir -p $NNSSL_ROOT/jsons
  
  # Prepare JSONs (80/20 split on train, val.json as test)
  $PYTHON_BIN $NNSSL_ROOT/src/nnssl/utilities/prepare_ped_v3_json_nnssl.py \
      --json-in $JSON_V3_REAL --val-json $JSON_ORIG_VAL --output-json $NNSSL_JSON_REAL
  
  $PYTHON_BIN $NNSSL_ROOT/src/nnssl/utilities/prepare_ped_v3_json_nnssl.py \
      --json-in $JSON_V3_MIXED --val-json $JSON_ORIG_VAL --output-json $NNSSL_JSON_MIXED

  OUT_REAL=$NNSSL_ROOT/training_runs/v3_mae/ped_real_fold${FOLD}
  OUT_MIXED=$NNSSL_ROOT/training_runs/v3_mae/ped_mixed_fold${FOLD}
  mkdir -p $OUT_REAL $OUT_MIXED

  echo "Training Real on GPU 0..."
  CUDA_VISIBLE_DEVICES=0 stdbuf -oL $PYTHON_BIN $NNSSL_ROOT/src/nnssl/evaluation/segmentation3d_ped_v3.py \
      --datalist-path $NNSSL_JSON_REAL \
      --pretrained-weights $PRETRAINED \
      --epochs 150 \
      --epoch-length 24 \
      --batch-size 2 \
      --learning-rate 0.0003 \
      --image-size 96 \
      --output-dir $OUT_REAL > $NNSSL_ROOT/sbatch/output/ped/real_fold${FOLD}.log 2>&1 &

  echo "Training Mixed on GPU 1..."
  CUDA_VISIBLE_DEVICES=1 stdbuf -oL $PYTHON_BIN $NNSSL_ROOT/src/nnssl/evaluation/segmentation3d_ped_v3.py \
      --datalist-path $NNSSL_JSON_MIXED \
      --pretrained-weights $PRETRAINED \
      --epochs 150 \
      --epoch-length 24 \
      --batch-size 2 \
      --learning-rate 0.0003 \
      --image-size 96 \
      --output-dir $OUT_MIXED > $NNSSL_ROOT/sbatch/output/ped/mixed_fold${FOLD}.log 2>&1

  wait
  echo "========== Finished NNSSL Fold ${FOLD} =========="
done
