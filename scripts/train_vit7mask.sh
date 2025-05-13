set -ex
/usr/bin/python3.11 /win/scallop/user/aditya/Pix2Pix/PyTorch-Pix2Pix-Modified/modified_train.py \
    --dataset_mode reconstruction \
    --dataroot "/win/scallop/user/aditya/PelvisRongensDataset/dataset_split" \
    --name Vit7Mask_ViKo_256 \
    --model vit7mask \
    --netG vitkonv_256 \
    --display_id  -1 \
    --display_freq 1 \
    --n_epochs  512 --n_epochs_decay 128 \
    --load_size 256 --crop_size 256 \
    --input_nc 1    --output_nc 1 \
    --patch_size 8 \
    --latent_size 2048 \
    --encoder_depth 9 \
    --total_heads 12 \
    --vit_ff_size 1024 \
    --batch_size 16 \
    --lr 5e-4 \
    --visible_patches 0.3