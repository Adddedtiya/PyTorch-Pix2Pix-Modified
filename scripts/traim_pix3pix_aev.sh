# set -ex
/usr/bin/python3.11 /win/scallop/user/aditya/Pix2Pix/PyTorch-Pix2Pix-Modified/modified_train.py \
    --dataset_mode reconstruction \
    --dataroot "/win/scallop/user/aditya/PelvisRongensDataset/dataset_split" \
    --name  pix3pix_aev_v4_512 --model pix3pix \
    --netG  ae_vec_v4 \
    --display_id  -1 \
    --display_freq 1 \
    --n_epochs 512  --n_epochs_decay 128 \
    --load_size 256 --crop_size 256 \
    --input_nc 1    --output_nc 1 \
    --batch_size 16 \
    --lr 5e-4