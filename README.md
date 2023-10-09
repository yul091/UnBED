# UnBED
Codebase for the ACL 2023 paper: "Uncertainty-Aware Bootstrap Learning for Joint Extraction on Distantly-Supervised Data" ([PDF](https://aclanthology.org/2023.acl-short.116.pdf))


### Quick Start

- Python 3.8+
- Install requirements: `pip install -r requirements.txt`


- Train and evaluate a joint extraction model with uncertainty-aware instance selection
```
MODEL_PATH=gpt2-medium
TRAIN_FILE="path/to/your/data"
VAL_FILE="path/to/your/data"
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
``` 

- Train and evaluate a joint extraction model with standard training (without instance selection)
```
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
    --overwrite_output_dir \
    --baseline
``` 



### Citation
If you find this repo useful, please cite our paper:
```bibtex
@inproceedings{li2023uncertainty,
  title={Uncertainty-Aware Bootstrap Learning for Joint Extraction on Distantly-Supervised Data},
  author={Li, Yufei and Yu, Xiao and Liu, Yanchi and Chen, Haifeng and Liu, Cong},
  booktitle={Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2023}
}
```
