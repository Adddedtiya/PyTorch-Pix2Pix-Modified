#! /bin/bash

# project information
project_dir="/win/scallop/user/aditya/Pix2Pix/PyTorch-Pix2Pix-Modified"
simg_path="/win/scallop/user/aditya/SingularityImage/compiled_images/aditya.scallop.naist_baseline-torch_3.0-2025-03-20-c2240002703f.sif"
data_root="/win/scallop/user/aditya/PelvisRongensDataset"
num_core=8
num_gpu=1
node="cl-yamaneko"

# Create output directory
slurm_out="/win/scallop/user/aditya/Pix2Pix/slurm_out"

#EXECUTE
sbatch \
    --no-requeue \
    --gres=gpu:${num_gpu} \
    -n ${num_core} \
    -D ${project_dir} \
    -w ${node}  \
    -o "${slurm_out}/%j_${node}.out" \
    --wrap="singularity exec --nv -B ${project_dir},${data_root} ${simg_path} /bin/bash /win/scallop/user/aditya/Pix2Pix/PyTorch-Pix2Pix-Modified/scripts/train_pix2pix_mask.sh"  