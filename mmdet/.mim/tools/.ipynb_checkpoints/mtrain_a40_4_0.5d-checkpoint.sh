#!/bin/bash
#SBATCH --qos bham
#SBATCH --account ninicj-sam-image
#SBATCH --time 0-11:59:59
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 36
#SBATCH --constraint=a100_40
#SBATCH --gpus-per-node 4

#SBATCH --mail-type=ALL
#SBATCH --mail-user=zxy239@student.bham.ac.uk

module purge
module load baskerville
module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0

source ~/use_mmdet2.sh

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/train.py "$@"\
    --launcher pytorch