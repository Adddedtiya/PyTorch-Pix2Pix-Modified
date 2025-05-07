# set -ex
/usr/bin/python3.11 /win/scallop/user/aditya/Pix2Pix/PyTorch-Pix2Pix-Modified/modified_train.py \
    --dataset_mode reconstruction \
    --dataroot "/win/scallop/user/aditya/PelvisRongensDataset/dataset_split" \
    --name pix3pix_mobilenetv4_binary_vector_512 --model pix3pix \
    --netG ae_vec_mv4 --blocks_count 5 --blocks_ratio 2 \
    --timm "mobilenetv4_conv_small.e2400_r224_in1k" \
    --display_id  -1 \
    --display_freq 1 \
    --num_threads 4 \
    --n_epochs 512  --n_epochs_decay 128 \
    --load_size 256 --crop_size 256 \
    --input_nc 1    --output_nc 1 \
    --batch_size 16 \
    --lr 5e-4