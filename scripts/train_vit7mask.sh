set -ex
/usr/bin/python3.11 /win/scallop/user/aditya/Pix2Pix/PyTorch-Pix2Pix-Modified/modified_train.py \
    --dataset_mode reconstruction \
    --dataroot "/win/scallop/user/aditya/PelvisRongensDataset/dataset_split" \
    --name Vit7Mask_256_symmetrical \
    --model vit7mask \
    --netG simvit_256 \
    --display_id  -1 \
    --display_freq 1 \
    --n_epochs 256  --n_epochs_decay 128 \
    --load_size 256 --crop_size 256 \
    --input_nc 1    --output_nc 1 \
    --patch_size 32 \
    --encoder_depth 6 \
    --decoder_depth 6 \
    --batch_size 128