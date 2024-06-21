#!/usr/bin/zsh
#SBATCH --job-name=tweety_sft            # Job name
#SBATCH --output=logs/%j.out             # Output file
#SBATCH --error=logs/%j.err              # Error file
#SBATCH --partition=boost_usr_prod                  # Specify the partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (processes) per node
#SBATCH --cpus-per-task=12
#SBATCH --qos=boost_qos_lprod
#SBATCH --gpus-per-node=1                # Number of tasks (processes) per node
#SBATCH --time=32:00:00                   # Walltime limit (hh:mm:ss)

export WANDB_MODE=offline
export WANDB_PROJECT="tiktotok-nl"
export WANDB_GROUP="mistral-7b"
export WANDB_JOB_TYPE="sft"
export TOKENIZERS_PARALLELISM=false

source ~/.zshrc
source $FAST/handbook/bin/activate

ACCELERATE_LOG_LEVEL=info srun accelerate launch \
    --config_file recipes/accelerate_configs/multi_gpu.yaml \
    --num_processes=1 scripts/run_sft.py \
    recipes/llama3-tweety-8b-italian/sft/config_qlora_tagengo.yaml

deactivate
