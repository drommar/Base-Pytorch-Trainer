cd ./src/main/ && \
python ./predict.py \
--config_file_path ../../configs/eval_config.yaml \
--model_type efficient_model \
--dataset_dir ../../dataset/ \
--submission_file_path ../../submissions/submission.csv