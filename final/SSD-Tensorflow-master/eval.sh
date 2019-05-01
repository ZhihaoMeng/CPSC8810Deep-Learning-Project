DATASET_DIR=./VOC2012/test_tf
EVAL_DIR=./logs/ssd_300_vgg_eval
CHECKPOINT_PATH=./logs/ssd_300_vgg_train/model.ckpt-477083
python3 eval_ssd_network.py \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2012\
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --wait_for_checkpoints=False \
    --batch_size=1 \
    --max_num_batches=100 \
