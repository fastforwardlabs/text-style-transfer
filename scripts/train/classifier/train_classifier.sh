#!/bin/bash

# This script should be run from the /home/cdsw directory
# Run classifier training script by passing a model name as argument
# EX: ./train_classifier.sh <model-name>
model_name=$1

echo "Training new model that will be saved at ~/models/$model_name"

ldconfig

python3 scripts/train/classifier/train_classifier.py \
    --model_name_or_path "bert-base-uncased" \
    --output_dir="./models/$model_name" \
    --overwrite_output_dir=True \
    --learning_rate=3e-05 \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --num_train_epochs=5 \
    --logging_dir="./models/logs/$model_name" \
    --logging_steps=2000 \
    --logging_strategy="steps" \
    --eval_steps=4000 \
    --evaluation_strategy="steps" \
    --save_total_limit=5 \
    --save_steps=4000 \
    --load_best_model_at_end=True \
    --metric_for_best_model="eval_accuracy" \
    --greater_is_better=True \

cp ./scripts/train/classifier/train_classifier.sh "./models/$model_name/COPY-train_classifier.sh"
echo "Placed a copy of this train script in ./models/$model_name/COPY-train_classifier.sh"