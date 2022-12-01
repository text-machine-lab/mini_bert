import math
import random
import argparse
import logging
import wandb
import utils
import time
import torch
import transformers
import numpy as np
import os
from pathlib import Path
from functools import partial
from tqdm.auto import tqdm
from LMTrainer import LMTrainer

"""
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
"""


def parse_args():
    """This function creates argument parser and parses the scrip input arguments.
    This is the most common way to define input arguments in python.

    To change the parameters, pass them to the script, for example:

    python cli/train.py \
        --output_dir output_dir \
        --weight_decay 0.01

    """
    parser = argparse.ArgumentParser(
        description="Train machine translation transformer model"
    )

    # Required arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./output_dir',
        #required=True,
        help=(
            "Where to store the final model. "
            "Should contain the source and target tokenizers in the following format: "
            r"output_dir/{source_lang}_tokenizer and output_dir/{target_lang}_tokenizer. "
            "Both of these should be directories containing tokenizer.json files."
        ),
    )
    parser.add_argument(
        "--restart",
        default=False,
        action="store_true",
        help="If experiment was stopped and needs to be restarted",
    )
    parser.add_argument(
        "--restart_for_fine_tuning",
        default=False,
        action="store_true",
        help="If experiment was stopped and needs to be restarted",
    )

    # Data arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./../../../from_shala/vocabulary_analysis/data_filtration/Filtration_15Nov2022/ALL_FILTERED_DATA/processed_data.json",
        help="path to raw dataset",
    )
    parser.add_argument(
        "--dataset_attribute",
        type=str,
        default="qnli",
        help="glue task to evaluate on",
    )
    
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.02,
        help="Size of the validation split of the pre-training data",
    )
    
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./tokenizer_selection_scripts/Tokenizer_files/roberta-base_17000",
        help="path to tokenizer.  If not provided, default BERT tokenizer will be used.",
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Whether to use a small subset of the dataset for debugging.",
    )
    
    parser.add_argument(
        "--fixed_seed_val",
        type=int,
        default=1,
        help="Value of the seed to use for data splitting and weights init",
    )

    # Model arguments
    parser.add_argument(
        "--masked_percent",
        default=0.15,
        type=float,
        help="Percentage of input to mask",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="The maximum total sequence length for source and target texts after "
             "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
             "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="Overwrite the cached training and evaluation sets",
    )

    # Training arguments
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu) on which the code should run",
    )
    parser.add_argument(
        "--device_index",
        type=str,
        default='0',
        help="which GPU to use",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=320,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of data items to use when debugging.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="highest learning rate value.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay value for AdamW optimizer",
    )
    
    parser.add_argument(
        "--glue_learning_rate",
        type=float,
        default=0.01,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Beta1 to use for Adam.",
    )
    
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="Beta2 to use for Adam.",
    )
    parser.add_argument(
        "--glue_beta2",
        type=float,
        default=0.999,
        help="Beta2 to use for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--glue_epochs",
        type=int,
        default=20,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--grad_acc_steps",
        type=int,
        default=1,
        help="Accumulate gradient for these many steps",
    )
    
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=200,
        help="Perform evaluation every n network updates.",
    )
    
    parser.add_argument(
        "--save_random_model",
        type=bool,
        default=True,
        help="save initial checkpoint of the model",
    )
    parser.add_argument(
        "--eval_random_model",
        type=bool,
        default=False,
        help="evaluate initial checkpoint of the model",
    )
    
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Compute and log training batch metrics every n steps.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=transformers.SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )

    parser.add_argument(
        "--wandb_project",
        default="mini_bert_fixed_seed",
        help="wandb project name to log metrics to",
    )

    args = parser.parse_args()

    return args



def main():
    args = parse_args()
    
    # fix seed
    torch.manual_seed(args.fixed_seed_val)
    np.random.seed(args.fixed_seed_val)
    random.seed(args.fixed_seed_val)
    
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # start wandb
    wandb.init(project=args.wandb_project, config=args)
    
    # make sure output dir exists
    args.output_dir = os.path.join(args.output_dir, wandb.run.name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # init everything (tokenizer, dataloader, model, criterion, optimizer and scheduler)
    LM_trainer = LMTrainer(args)
    print(f'Size of the model is: {LM_trainer.model_size}')
    
    # save random model for testing performance on random model
    if args.save_random_model:
        os.makedirs(os.path.join(args.output_dir, 'random_model'))
        LM_trainer.model.save_pretrained(os.path.join(args.output_dir, 'random_model'))
    
    # train the model
    if not args.eval_random_model:
        # start training
        trained_model = LM_trainer.train_model(args)
    else:
        trained_model = LM_trainer.model
    
    
    # evaluation on GLUE
    # -log GLUE metrics on wandb
    
    
    #
    wandb.finish()
    #logger.info(f"***** FINSHED TRAINING AND EVAL *****")
    
    return


if __name__ == "__main__":
    main()

# python3 train.py --beta2=0.95 --learning_rate=0.00005 --max_train_steps=1 --restart --output_dir=output_dir/dazzling-haze-202 --tokenizer_path=Sentence_13k --batch_size=10 --glue_learning_rate=0.01 --glue_epochs=100 --restart_for_fine_tuning
