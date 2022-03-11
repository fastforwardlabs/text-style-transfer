#!/bin/bash

# This script should be run from the /home/cdsw directory
# Run TST training script by passing a model name as argument
# EX: ./train.sh <model-name>
model_name=$1

echo "Training new model that will be saved at ~/models/$model_name"

ldconfig

python3 scripts/run_TST.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --do_predict \
    --dataset_name "wnc_one_word" \
    --num_train_epochs=10 \
    --output_dir "./models/$model_name" \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \
    --eval_steps=1000 \
    --predict_with_generate \
    --generation_max_length=1024 \
    --generation_num_beams=4 \
    --logging_strategy "steps" \
    --logging_steps=1000 \
    --logging_dir "./models/logs/$model_name" \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --save_total_limit=5 \
    --save_steps=1000 \
    --load_best_model_at_end=True \
    --metric_for_best_model "eval_accuracy" \
    --greater_is_better=True \
    # --max_train_samples=10000 \
    # --max_eval_samples=250 \
    # --max_predict_samples=250

cp ./scripts/train.sh "./models/$model_name/COPY-train.sh"
echo "Placed a copy of this train script in ./models/$model_name/COPY-train.sh"