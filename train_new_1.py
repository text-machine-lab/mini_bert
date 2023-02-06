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
import pandas as pd
import os
from pathlib import Path
from functools import partial
from tqdm.auto import tqdm
from LMTrainerNew import LMTrainer
import time
import json
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
        "--checkpoint_dir",
        type=str,
        default='./output_dir',#/earthy-moon-78',
        help="Where to find previous checkpoint",
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
        default="./pretraining_data_01Jan2022",
        help="path to raw dataset",
    )
    parser.add_argument(
        "--use_wiki_data",
        type=bool,
        default=False,
        help="Use wikipedia data instead of filtered data",
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
        default="./tokenizer_selection_scripts/Tokenizer_files/roberta-base_19000",#"./tokenizer_selection_scripts/Tokenizer_files_free_text/roberta-base_40000",
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
        default=0,
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
        "--num_hidden_layers",
        type=int,
        default=8,
        help="Number of hidden layers of transformer blocks",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=8,
        help="Number of attention heads in each of the transformer blocks",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="dimension of embeddings and hidden vectors",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=1024,
        help="dimension of intermediate hidden vectors for Q, K and V",
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
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of data items to use when debugging.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0005,
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
        default=0.98,
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
        default=4,
        help="Accumulate gradient for these many steps",
    )
    
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=4000,
        help="Perform evaluation every n network updates.",
    )
    
    parser.add_argument(
        "--save_checkpoint_evey_steps",
        type=int,
        default=4000,
        help="Save model checkpoint",
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
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_percent",
        type=float,
        default=0.05,
        help="Total number of training epochs to perform.",
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
        default="mini_bert_ACL",
        help="wandb project name to log metrics to",
    )
    
    #
    args, unknownargs = parser.parse_known_args()

    return args


def one_run(
    embedding_size=256,
    hidden_size=256,
    num_attention_heads=8,
    num_hidden_layers=8,
    intermediate_size=1024,
    tags=[],
):
    args = parse_args()
    
    #
    args.embedding_size = embedding_size
    args.hidden_size = hidden_size
    args.intermediate_size = intermediate_size
    args.num_attention_heads = num_attention_heads
    args.num_hidden_layers = num_hidden_layers
    
    # fix seed
    torch.manual_seed(args.fixed_seed_val)
    random.seed(args.fixed_seed_val)
    np.random.seed(args.fixed_seed_val)
    transformers.set_seed(args.fixed_seed_val)
    
    # set device
    #os.environ["CUDA_VISIBLE_DEVICES"] = '1'#args.device_index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # start wandb
    wandb.init(
        project=args.wandb_project, 
        config=args,
        tags=tags,
    )
    
    # make sure output dir exists
    args.output_dir = os.path.join(args.output_dir, wandb.run.name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.run_name = wandb.run.name
    
    
    # init everything (tokenizer, dataloader, model, criterion, optimizer and scheduler)
    LM_trainer = LMTrainer(args)
    print(f'Size of the model is: {LM_trainer.model_size}')
    
    """
    # @TODO: delete this when done
    for name, module in LM_trainer.model.named_modules():
        if 'emb2hidden' == name:
            print(f"\n{name}")
            print(module)
        
        if "embeddings" == name:
            print(f"\n{name}")
            print(module)
        
        #if 'layer.0' in name:
        #    print(f"\n{name}")
        #    print(module)
    """
    
    # save random model for testing performance on random model
    if args.save_random_model:
        os.makedirs(os.path.join(args.output_dir, 'random_model'))
        LM_trainer.model.save_pretrained(os.path.join(args.output_dir, 'random_model'))
    
    # train the model
    metrics = LM_trainer.train_model(args)
    
    # evaluation on GLUE
    # -log GLUE metrics on wandb
    
    
    #
    wandb.finish()
    #logger.info(f"***** FINSHED TRAINING AND EVAL *****")
    
    #
    metrics['run_name'] = args.run_name
    
    return metrics

def start_experiment():
    
    #
    timestamp_ = int(time.time())
    date = "14Jan2023"
    
    #
    features_to_vary = {
        #'embedding_size': [16, 8],
        #'hidden_size': [32, 16, 8],
        #'num_hidden_layers': [2, 1],#[4, 2, 1],
        'intermediate_size': [128, 64, 32],
        #'num_attention_heads': [4, 2, 1],
    }
    total_runs = sum([features_to_vary[k_].__len__() for k_ in features_to_vary])
    
    # to save test results
    test_results = pd.DataFrame(
        -1,
        index=range(total_runs),
        columns=[
            "run number",
            "embedding_size",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_hidden_layers",
            "Embedding parameters",
            "Non-embedding parameters",
            "Total parameters",
        ],
    )
    
    # to save eval results
    eval_results = pd.DataFrame(
        -1,
        index=range(total_runs * 36000),
        columns=[
            "eval/perplexity",
            "eval/loss",
            "eval/step",
            "eval/epoch",
            "eval/batch_idx",
            "eval/updates",
        ],
    )    
    
    #
    run_idx = -1
    for feature in features_to_vary:        
        for feature_val in features_to_vary[feature]:
            
            #
            run_idx += 1
            
            #
            input_config = {
                "embedding_size": 64,
                "hidden_size": 64,
                "intermediate_size": 256,
                "num_attention_heads": 8,
                "num_hidden_layers": 8,
            }
            input_config[feature] = feature_val
            input_config["tags"] = [feature, "ModelConfig"]
            
            #
            print(f"\nStarting run with following configuration")
            print(input_config)
            print('\n')
            metrics = one_run(**input_config)
            input_config["tags"] = ', '.join(input_config["tags"])
            
            # save input configuration
            for k_, v_ in input_config.items():
                test_results.loc[run_idx, k_] = v_
            
            # save test results
            for k_, v_ in metrics.items():
                if not 'eval/' in k_:
                    test_results.loc[run_idx, k_] = v_
            
            # save eval results
            for k_, v_ in metrics.items():
                if 'eval/' in k_:
                    len_ = len(v_)
                    start = run_idx * len_
                    end = start + len_ - 1
                    eval_results.loc[start:end, k_] = v_
                else:
                    len_ = len(metrics['eval/perplexity'])
                    start = run_idx * len_
                    end = start + len_ - 1
                    eval_results.loc[start:end, k_] = [v_] * len_
                    
            for k_, v_ in input_config.items():
                eval_results.loc[start:end, k_] = [v_] * len_
            
            
            # save results
            try:
                # save
                test_results.to_csv(
                    os.path.join(
                        ".",
                        "CSV files with experiment results",
                        f"ModelConfig_{date}",
                        f"experiment_results_test_{timestamp_}_{feature}.csv"
                    )
                )
                eval_results.to_csv(
                    os.path.join(
                        ".",
                        "CSV files with experiment results",
                        f"ModelConfig_{date}",
                        f"experiment_results_eval_{timestamp_}_{feature}.csv"
                    )
                )
            except:
                test_results.to_csv(f"FOLDER_NOT_FOUND_experiment_results_test_{timestamp_}_{feature}.csv")
                eval_results.to_csv(f"FOLDER_NOT_FOUND_experiment_results_eval_{timestamp_}_{feature}.csv")
            
            
            # update map
            try:
                with open("map_.json", "r") as f:
                    map_ = json.load(f)
                map_[metrics["run_name"]] = input_config
                with open("map_.json", "w") as f:
                    json.dump(map_, f, indent=4)
            except:
                map_ = {}
                map_[metrics["run_name"]] = input_config
                with open(f"map_{timestamp_}_{feature}.json", "w") as f:
                    json.dump(map_, f, indent=4)
                
            
    return

def start_experiment_isoflops():
    
    #
    timestamp_ = int(time.time())
    
    # every tuple should be = (embedding_size, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers)
    all_experiments = [
        # ISO-FLOP
        #(32, 32, 128, 2, 2),
        #(32, 32, 256, 2, 1),
        #(64, 128, 1024, 8, 4),
        #(128, 128, 128, 1, 1),
        #(128, 32, 256, 2, 2),
        #(128, 32, 512, 8, 1),
        
        # ISO-PR
        #(64, 32, 128, 1, 8),
        #(64, 32, 256, 1, 4),
        #(256, 128, 128, 4, 4),
        #(256, 128, 512, 8, 1),
        
        #(128, 128, 256, 1, 2),
        #(64, 64, 512, 8, 2),
        #(64, 64, 1024, 8, 1),
        
        #(32, 64, 256, 4, 8),
        (64, 128, 128, 2, 8),
        (64, 128, 512, 8, 4),
        (128, 256, 1024, 1, 2),
        
    ]
    total_runs = len(all_experiments)
    
    # to save test results
    test_results = pd.DataFrame(
        -1,
        index=range(total_runs),
        columns=[
            "run number",
            "embedding_size",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_hidden_layers",
            "Embedding parameters",
            "Non-embedding parameters",
            "Total parameters",
        ],
    )
    
    # to save eval results
    eval_results = pd.DataFrame(
        -1,
        index=range(total_runs * 30000),
        columns=[
            "eval/perplexity",
            "eval/loss",
            "eval/step",
            "eval/epoch",
            "eval/batch_idx",
            "eval/updates",
        ],
    )    
    
    #
    run_idx = -1
    for exp in all_experiments:        
        # embedding_size, hidden_sizem intermediate_size, num_attention_heads, num_hidden_layers
        (e_, h_, i_, a_, l_) = exp
            
        #
        run_idx += 1

        #
        input_config = {
            "embedding_size": e_,
            "hidden_size": h_,
            "intermediate_size": i_,
            "num_attention_heads": a_,
            "num_hidden_layers": l_,
        }

        #
        print(f"\nStarting run with following configuration")
        print(input_config)
        print('\n')
        metrics = one_run(**input_config)

        # save input configuration
        for k_, v_ in input_config.items():
            test_results.loc[run_idx, k_] = v_

        # save test results
        for k_, v_ in metrics.items():
            if not 'eval/' in k_:
                test_results.loc[run_idx, k_] = v_

        # save eval results
        for k_, v_ in metrics.items():
            if 'eval/' in k_:
                len_ = len(v_)
                start = run_idx * len_
                end = start + len_ - 1
                eval_results.loc[start:end, k_] = v_
            else:
                len_ = len(metrics['eval/perplexity'])
                start = run_idx * len_
                end = start + len_ - 1
                eval_results.loc[start:end, k_] = [v_] * len_

        for k_, v_ in input_config.items():
            eval_results.loc[start:end, k_] = [v_] * len_


        # save
        test_results.to_csv(f"experiment_results_test_{timestamp_}_ISO-PR_1.csv")
        eval_results.to_csv(f"experiment_results_eval_{timestamp_}_ISO-PR_1.csv")
    
    
    return


if __name__ == "__main__":
    start_experiment()
    #start_experiment_isoflops()

# python3 train.py --beta2=0.95 --learning_rate=0.00005 --max_train_steps=1 --restart --output_dir=output_dir/dazzling-haze-202 --tokenizer_path=Sentence_13k --batch_size=10 --glue_learning_rate=0.01 --glue_epochs=100 --restart_for_fine_tuning
