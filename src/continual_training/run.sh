#!/usr/bin/zsh
#SBATCH --job-name=train_bert            # Job name

#SBATCH --error=logs/%j.err              # Error file
#SBATCH --partition=boost_usr_prod                  # Specify the partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8
#SBATCH --qos=boost_qos_lprod
#SBATCH --gpus-per-node=4                # Number of tasks (processes) per node
#SBATCH --time=4-00:00:00                   # Walltime limit (hh:mm:ss)
#SBATCH --mem-per-gpu=64G

export WANDB_MODE=offline
export WANDB_PROJECT="tiktotok-nl"
export WANDB_JOB_TYPE="continual-learning"
export TOKENIZERS_PARALLELISM=false

source ~/.zshrc
source $FAST/tiktotok/bin/activate

BASE_MODEL="Tweeties/tweety-7b-italian-v24b-llama3"
NEW_MODEL="g8a9/llama3-tweety-8b-italian"

module load cuda

srun accelerate launch --multi_gpu --mixed_precision=bf16 train.py \
    --base_model ${BASE_MODEL} \
    --context_length 4096 \
    --dataset_name='gsarti/clean_mc4_it' \
    --dataset_subname='full' \
    --new_model_name ${NEW_MODEL} \
    --output_dir $FAST/llama3-tweety-8b-italian \
    --batch_size 1 \
    --gradient_accumulation_steps 128

deactivate
