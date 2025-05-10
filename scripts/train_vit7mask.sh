set -ex
/usr/bin/python3.11 /win/scallop/user/aditya/Pix2Pix/PyTorch-Pix2Pix-Modified/modified_train.py \
    --dataset_mode reconstruction \
    --dataroot "/win/scallop/user/aditya/PelvisRongensDataset/dataset_split" \
    --name Vit7Mask_256 \
    --model vit7mask \
    --netG simvit_256 \
    --display_id  -1 \
    --display_freq 1 \
    --n_epochs 512  --n_epochs_decay 128 \
    --load_size 256 --crop_size 256 \
    --input_nc 1    --output_nc 1 \
    --patch_size 16 \
    --encoder_depth 11 \
    --decoder_depth 17 \
    --batch_size 16
