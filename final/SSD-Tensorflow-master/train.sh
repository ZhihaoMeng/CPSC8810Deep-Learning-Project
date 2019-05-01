DATASET_DIR=./VOC2012/trainval_tf
TRAIN_DIR=./logs/ssd_300_vgg_train
CHECKPOINT_PATH=./checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2012 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH}\
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.0001 \
    --batch_size=8
