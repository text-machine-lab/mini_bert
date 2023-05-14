
# Honey, I Shrunk the Language: Language Model Behavior at Reduced Scale
In this repository, we present the pre-training data and codebase used in our study (LINK) presented at ACL 2023. We study effect of reduction in the vocabulary of the language on the pre-training and NLU performance of language models with less than 20 million parameters.

## Key Findings

1. Smaller models start showing pre-training benefits earlier if simplified language data (smaller vocabulary) is used to train the models.
2. In the downscaled setting, 
    - we observe a break in the power curve between FLOPs and Evaluation Loss
    - compute-optimality is not crucial
    - pure parameter is count is not predictive of the pre-training performance


## Important Files

1. `./data/vocabulary/AOChildes_word_frequency`: This file includes the vocabulary we focus on. The vocabulary is curated from the AOChildes (LINK) dataset which consists of transcripts of child-directed (younger than 6 years old) speech. We use this vocabulary to collect pre-training data for our experiments.


2. `./data/pretraining_data/constrained_language`: This directory consists of the pre-training data we used for our main experiments. All text sequences in the data are strictly restricted to the words contained in the pre-defined vocabulary `./data/vocabulary/AOChildes_word_frequency`. To curate the dataset, we filtered text sequences from five open text corpora namely, C4, BookCorpus, Wikipedia, Simplified-Wikipedia and Children's Book Test corpus. The dataset consists of $\approx$ 9 million training sequences ($\approx$ 1.1 billion tokens) and 100,000 validation and test sequences.

3. `./data/pretraining_data/unconstrained_language`: This directory consists of text sequences that are not constrained by any vocabulary rule. We use this dataset for our supplementary experiments on unconstrained language data. The size of the dataset and the distribution over various copora are approximately matched to the constrained language data `./data/pretraining_data/constrained_language`. The validation and text splits of the data are kept exactly same as in `./data/pretraining_data/constrained_language`, for comparability.


## Installation and Training Models

For replicating our experiments, please run following commands.

1. Creating a vitual environment

`conda create -n mini_bert python=3.7`
`conda activate mini_bert`
`pip install -editable ./`


2. Pre-training models

Here, in addition to the regular hyperparameters, the users have flexibility of selecting configuration hyperparameters of their choice. Particulary, we have functionality to set embedding dimension, hidden dimension in the transformer block, the intermediate dimension of the ffn in the transformer block, and number of layers of transformer block. We provide a sample pre-training initialization command,

`enter command here`

Furthermore, the functins `vary_configuration()` and `train_a_set_of_configurations()` in the `start_pretraining.py` file can be used to train a set of different model configurations sequentially.


2. Fine-tuning models

For measuring NLU capabilities we fine-tune the pre-trained models on GLUE tasks. We adpat the `run_glue.py` script provided by Huggingface (at https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) to our experimental setting. We add one argument `--filter_glue` to the original script in order accomodate fine-tuning on both constrained and unconstrained version of the GLUE datasets. We provide a sample fine-tuning initialization command,

`enter command here`
