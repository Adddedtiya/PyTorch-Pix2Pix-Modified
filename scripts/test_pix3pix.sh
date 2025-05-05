set -ex
/usr/bin/python3 /win/scallop/user/aditya/Pix2Pix/PyTorch-Pix2Pix-Modified/modified_test.py \
    --name pix3pix_aev_512 \
    --dataset_mode reconstruction \
    --epoch best \
    --dataroot "/win/scallop/user/aditya/PelvisRongensDataset/dataset_split" \
    --model pix3pix \
    --netG  ae_vec \
    --load_size 256 --crop_size 256 \
    --input_nc 1    --output_nc 1

