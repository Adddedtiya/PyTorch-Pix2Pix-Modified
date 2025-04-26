set -ex
/usr/bin/python3.11 /win/scallop/user/aditya/Pix2Pix/PyTorch-Pix2Pix-Modified/modified_train.py \
    --dataset_mode curtain \
    --dataroot "/win/scallop/user/aditya/PelvisRongensDataset/dataset_split" \
    --name  pix7mask_aot9_512 \
    --model pix7mask \
    --netG  ae_9aot \
    --display_id  -1 \
    --display_freq 1 \
    --n_epochs 512  --n_epochs_decay 128 \
    --load_size 256 --crop_size 256 \
    --input_nc 2    --output_nc 1 \
    --batch_size 32
