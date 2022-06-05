if [ ! -d "checkpoints" ];then
  mkdir checkpoints;
fi
cd ./src/main/ && \
python ./train.py \
--config_file_path ../../configs/train_config.yaml \
--epochs 500 \
--batch_size 8 \
--dataset_dir ../../dataset/ \
--model_type efficient_model \
--checkpoints_dir ../../checkpoints/ \
| tee ../../checkpoints/output.txt