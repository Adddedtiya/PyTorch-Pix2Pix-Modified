set -ex
/usr/bin/python3.11 /win/scallop/user/aditya/Pix2Pix/PyTorch-Pix2Pix-Modified/train.py \
    --dataset_mode masking \
    --dataroot "/win/scallop/user/aditya/PelvisRongensDataset/dataset_split" \
    --mask_root "/win/scallop/user/aditya/PelvisRongensDataset/base_masks" \
    --name mdf_r9_px256_t521 \
    --model pix2pix \
    --direction AtoB \
    --netG resnet_9blocks \
    --display_id -1 \
    --n_epochs 512 --n_epochs_decay 128 \
    --load_size 256 --crop_size 256 \
    --input_nc 2 --output_nc 1 \
    --batch_size 32
