#!/bin/bash

# -------------------------------
# GEMMA-3-12B CPT + SFT Training
# -------------------------------

# 指定要使用的 GPU，例如 6,7,8,9
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# 訓練腳本名稱
TRAIN_SCRIPT="cpt_sft_ex_.py"

# nohup 輸出檔案
NOW=$(date +"%m%d_%H%M")
LOG_FILE="train_gemma3_12b_$NOW.log"

echo "🚀 Starting GEMMA-3-12B CPT + SFT training..."
echo "📌 Logging to $LOG_FILE"
echo "📌 Using GPUs: $CUDA_VISIBLE_DEVICES"

# 執行訓練，nohup + 背景執行
nohup python $TRAIN_SCRIPT > $LOG_FILE 2>&1 &

# 顯示背景進程資訊
echo "✅ Training launched with PID: $!"
echo "You can check progress with: tail -f $LOG_FILE"
