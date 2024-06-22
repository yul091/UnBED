MODEL_PATH=gpt2-medium
TRAIN_FILE="./datasets/nyt-h/train_nonna.json"
VAL_FILE="./datasets/nyt-h/dev.json"
MODEL_UNCERTAINTY="prob_variance" # bald, max_prob, prob_variance
DATA_UNCERTAINTY="entropy" # vanilla, entropy
COMMITTEE_SIZE=10
THRESHOLD=0.6
ENSEMBLE_WEIGHT=1.0
OUTPUT_DIR="results//NYT_${MODEL_UNCERTAINTY}_${THRESHOLD}_${DATA_UNCERTAINTY}_${COMMITTEE_SIZE}"

python run_uncertainty.py \
    --model_name_or_path $MODEL_PATH \
    --classifier_type "crf" \
    --train_file $TRAIN_FILE \
    --validation_file $VAL_FILE \
    --output_dir $OUTPUT_DIR \
    --do_eval \
    --do_train \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model eval_f1 \
    --greater_is_better True \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 20 \
    --gradient_accumulation_steps 16 \
    --overwrite_cache \
    --use_negative_sampling \
    --sample_rate 0.1 \
    --num_train_epochs 100 \
    --boot_start_epoch 5 \
    --threshold $THRESHOLD \
    --committee_size $COMMITTEE_SIZE \
    --ensemble_weight $ENSEMBLE_WEIGHT \
    --save_total_limit 2 \
    --model_uncertainty $MODEL_UNCERTAINTY \
    --data_uncertainty $DATA_UNCERTAINTY \
    --overwrite_output_dir
