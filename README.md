# Tweety Ita Resources

This repository contains scripts and resources to replicate the training of Tweety Italian models.

The `src` folder contains python and bash script organized into:
- `continual_training`: to run a small number of adaptation steps in Italian after the tokenizer swap;  
- `alignment`: scripts and recipes to run SFT and DPO with HF's [alignment-notebook](https://github.com/huggingface/alignment-handbook)
- `datasets`: code to create dataset resources