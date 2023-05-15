# Honey, I Shrunk the Language: Language Model Behavior at Reduced Scale

In this repository, we present the pre-training data and codebase used in our study (LINK) presented at ACL 2023. We study effect of reduction in the vocabulary of the language on the pre-training and NLU performance of language models with less than 20 million parameters.

## Key Findings

<p align="center">
    <img 
         src="/results/ForACL/_ForACL_FLOPS Hoffman total_vs_eval-loss_w_power_curve_.png" 
         alt="Break in the power law" 
         title="Break in the power law"
         width="50%" 
         height="50%">
</p>

1. Smaller models start showing pre-training benefits earlier if simplified language data (smaller vocabulary) is used to train the models.
2. In the downscaled setting, 
    - we observe a break in the power curve between FLOPs and Evaluation Loss (`can add figure 1`)
    - compute-optimality is not crucial for NLU capabilities
    - pure parameter is count is not predictive of the pre-training performance


## Important Files

1. `./data/vocabulary/AOChildes_word_frequency`: This file includes the vocabulary we focus on. The vocabulary is curated from the AOChildes (https://github.com/UIUCLearningLanguageLab/AOCHILDES) dataset which consists of transcripts of child-directed speech. We use this vocabulary to collect pre-training data for our experiments.


2. `./data/pretraining_data/constrained_language`: This directory consists of the pre-training data we used for our main experiments. All text sequences in the data are strictly restricted to the words contained in the pre-defined vocabulary `./data/vocabulary/AOChildes_word_frequency`. To curate the dataset, we filtered text sequences from five open text corpora namely, C4, BookCorpus, Wikipedia, Simplified-Wikipedia and Children's Book Test corpus. The dataset consists of $\approx$ 9 million training sequences ($\approx$ 1.1 billion tokens) and 100,000 sequences for each, validation and test split.

3. `./data/pretraining_data/unconstrained_language`: This directory consists of text sequences that are not constrained by any vocabulary rule. We use this dataset for our supplementary experiments on unconstrained language data. The size of the dataset and the distribution over various copora are approximately matched to the constrained language data `./data/pretraining_data/constrained_language`. The validation and text splits of the data are kept exactly same as in `./data/pretraining_data/constrained_language`, for comparability.


## Installation and Training Models

For replicating our experiments, please run following commands.

1. Creating a vitual environment

```
conda create -n mini_bert python=3.7
conda activate mini_bert
pip install -editable ./
```


2. Pre-training models

Here, in addition to the regular hyperparameters, the users have flexibility of selecting configuration hyperparameters of their choice. Particulary, we have functionality to set embedding dimension, hidden dimension in the transformer block, the intermediate dimension of the ffn in the transformer block, and number of layers of transformer block. We provide a sample pre-training initialization command,

```
python scripts/start_pretraining.py --dataset_path="./data/pretraining_data/unconstrained_language" --tokenizer_path="./data/trained_tokenizers/tokenizer_files_unconstrained_language/roberta-base_31000" --masked_percent=0.10 --embedding_size=32 --hidden_size=32 --intermediate_size=32 --num_attention_heads=4 --num_hidden_layers=2 --batch_size=16 --eval_batch_size=16 --learning_rate=0.0001 --warmup_percent=0.05 --weight_decay=0.01 --beta1=0.9 --beta2=0.95 --num_train_epochs=1 --grad_acc_steps=8 --eval_every_steps=8000 --save_random_model=8000 --wandb_project="mini_bert_ACL_debug"

```

The functins `vary_configuration()` and `train_a_set_of_configurations()` in the `start_pretraining.py` file can be used to train a set of different model configurations sequentially. Furthermore, please note that in our experiments we utilized tokenizer at `./data/trained_tokenizers/tokenizer_files_constrained_language/roberta-base_19000` for our main experiments with the constrained language data and `./data/trained_tokenizers/tokenizer_files_unconstrained_language/roberta-base_31000` for the supplementary experiments with the unconstrained language data.


2. Fine-tuning models

For measuring NLU capabilities we fine-tune the pre-trained models on GLUE tasks. We adpat the `run_glue.py` script provided by Huggingface (at https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) to our experimental setting. We add one argument `--filter_glue` to the original script in order accomodate fine-tuning on both constrained and unconstrained version of the GLUE datasets. We provide a sample fine-tuning initialization command,

```
enter command here
```
