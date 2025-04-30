set -ex
/usr/bin/python3 /win/scallop/user/aditya/Pix2Pix/PyTorch-Pix2Pix-Modified/modified_test.py \
    --name pix7mask_cte9_512 \
    --dataset_mode curtain \
    --curtain_type start \
    --curtain_size 0.5 \
    --epoch best \
    --dataroot "/win/scallop/user/aditya/PelvisRongensDataset/dataset_split" \
    --model pix7mask \
    --netG ae_9cte \
    --load_size 256 --crop_size 256 \
    --input_nc 2    --output_nc 1

