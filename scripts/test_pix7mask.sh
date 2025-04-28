set -ex
/usr/bin/python3 /win/scallop/user/aditya/Pix2Pix/PyTorch-Pix2Pix-Modified/modified_test.py \
    --name pix7mask_resnet9_512 \
    --dataset_mode curtain \
    --dataroot "/win/scallop/user/aditya/PelvisRongensDataset/dataset_split" \
    --model pix7mask \
    --netG resnet_9blocks \
    --load_size 256 --crop_size 256 \
    --input_nc 2    --output_nc 1

