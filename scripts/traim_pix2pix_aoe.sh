set -ex
/usr/bin/python3.11 /win/scallop/user/aditya/Pix2Pix/PyTorch-Pix2Pix-Modified/modified_train.py \
    --dataset_mode reconstruction \
    --dataroot "/win/scallop/user/aditya/PelvisRongensDataset/dataset_split" \
    --name  pix2pix_ivn4_1024 --model pix2pix \
    --netG  aoe_ivg \
    --display_id  -1 \
    --display_freq 1 \
    --n_epochs 1024 --n_epochs_decay 128 \
    --load_size 256 --crop_size 256 \
    --input_nc 1    --output_nc 1 \
    --batch_size 16
