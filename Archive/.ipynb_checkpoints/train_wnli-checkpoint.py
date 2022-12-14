import copy
import math

import numpy
from torch import rand
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, BertTokenizer, \
    DataCollatorWithPadding, AutoModelForSequenceClassification

import random

from functools import partial

from torch.utils.data import DataLoader
import argparse
import torch

import transformers
import datasets
from datasets import load_from_disk, load_dataset, load_metric
import logging
from tqdm.auto import tqdm
import wandb
import utils
import json

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


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
        required=True,
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

    # Data arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="glue",
        help="path to raw dataset",
    )
    parser.add_argument(
        "--dataset_attribute",
        type=str,
        default="qnli",
        help="path to raw dataset",
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="path to tokenizer.  If not provided, default BERT tokenizer will be used.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mollypak/bert-model-baby",
        help="The name of model to be loaded. We will only take the model config.",
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Whether to use a small subset of the dataset for debugging.",
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
        "--batch_size",
        type=int,
        default=8,
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
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Beta2 to use for Adam.",
    )
    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout rate of the Transformer encoder",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=500,
        help="Perform evaluation every n network updates.",
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
        "--num_warmup_steps",
        type=int,
        default=1000,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--generation_type",
        choices=["greedy", "beam_search"],
        default="beam_search",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help=(
            "Beam size for beam search generation. "
            "Decreasing this parameter will make evaluation much faster, "
            "increasing this (until a certain value) would likely improve your results."
        ),
    )

    parser.add_argument(
        "--wandb_project",
        default="mini_bert_wnli_train",
        help="wandb project name to log metrics to",
    )

    args = parser.parse_args()

    return args


def evaluate_batch(model, batch, task):
    # turn on evaluation mode: no dropout
    metric = load_metric("glue", task)
    model.eval()
    average_loss = 0
    num_batchs = 0
    with torch.no_grad():
        model_output = model(**batch)

        logits = model_output.logits
        preds = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=preds, references=batch["labels"])
    model.train()

    return metric.compute()


def evaluate(model, eval_dataloader, device, task):
    # turn on evaluation mode: no dropout
    metric = load_metric("glue", task)
    model.eval()
    average_loss = 0
    num_batchs = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            model_output = model(**batch)

            average_loss += model_output.loss
            num_batchs += 1
            logits = model_output.logits
            preds = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=preds, references=batch["labels"])
    model.train()

    return metric.compute()


# import ipdb
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def make_dataloader(dataset, sentence1_key, sentence2_key, batch_size, data_collator, tokenizer):
    def tokenize_function(example):
        return tokenizer(example[sentence1_key], example[sentence2_key], truncation=True, max_length=128)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(['idx', sentence1_key, sentence2_key])
    dataset = dataset.rename_column('label', 'labels')
    dataset.set_format('pt')

    dataset = torch.utils.data.DataLoader(dataset,
                                          shuffle=True,
                                          batch_size=batch_size,
                                          collate_fn=data_collator
                                          )
    return dataset


def prep_dataset(tokenizer, dataset_attribute, batch_size, sample_size, debug=False):
    filtered_dataset = utils.filter_glue_dataset(dataset_name=dataset_attribute)

    if debug:
        filtered_dataset = utils.sample_small_debug_dataset(
            filtered_dataset, sample_size
        )

    train_dataset = filtered_dataset["train"]
    eval_dataset = filtered_dataset["validation"]
    print("Data loaded")
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = train_dataset.column_names
    print(f"Data column_names{column_names}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    sentence1_key, sentence2_key = task_to_keys[dataset_attribute]

    train_data = make_dataloader(train_dataset, sentence1_key, sentence2_key, batch_size, data_collator, tokenizer)
    eval_data = make_dataloader(eval_dataset, sentence1_key, sentence2_key, batch_size, data_collator, tokenizer)

    for batch in train_data:
        [print('{:>20} : {}'.format(k, v.shape)) for k, v in batch.items()]
        break
    return train_data, eval_data


def main():
    args = parse_args()
    wandb.init(project=args.wandb_project, config=args)

    device = args.device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path) if args.tokenizer_path else BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # read in data

    # raw_datasets = load_dataset(args.dataset_path, args.dataset_attribute)
    # print(f"dataset keys {raw_datasets.keys()}")
    train_data, eval_data = prep_dataset(tokenizer, args.dataset_attribute, args.batch_size, args.sample_size,
                                         args.debug)
    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, args.beta2)
    )

    num_update_steps_per_epoch = len(train_data)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = transformers.get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_data)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps))
    global_step = 0
    model.to(device)

    # Training loop
    for epoch in range(args.num_train_epochs):
        model.train()  # make sure that model is in training mode, e.g. dropout is enabled
        if global_step >= args.max_train_steps:
            break
        # iterate over batches
        for batch in train_data:
            if global_step >= args.max_train_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            # ipdb.set_trace()

            loss = model(**batch).loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            if global_step % args.logging_steps == 0:
                # An extra training metric that might be useful for understanding
                # how well the model is doing on the training set.
                # Please pay attention to it during training.
                # If the metric is significantly below 80%, there is a chance of a bug somewhere.

                wandb.log(
                    {
                        "fine_tuning_loss": loss,
                        "fine_tuing_learning_rate": optimizer.param_groups[0]["lr"],
                        "fine_tuing_epoch": epoch,
                    },
                    step=global_step,
                )

            if (
                    global_step % args.eval_every_steps == 0
                    or global_step == args.max_train_steps
            ):
                (
                    metric_acc
                ) = evaluate_batch(
                    model=model,
                    batch=batch,
                    task=args.dataset_attribute
                )

                wandb.log(
                    {
                        "eval/accuracy": metric_acc,
                    },
                    step=global_step,
                )
                if global_step >= args.max_train_steps:
                    break

                if global_step % args.eval_every_steps == 0:
                    metrics = evaluate(model, eval_data, device, task=args.dataset_attribute)
                    wandb.log(metrics, step=global_step)

                logger.info("Saving model checkpoint to %s", args.output_dir)
                model.save_pretrained(args.output_dir)

    logger.info("Final evaluation")
    metrics = evaluate(model, eval_data, device, args.dataset_attribute)
    wandb.log(metrics, step=global_step)

    logger.info("Saving final model checkpoint to %s", args.output_dir)
    model.save_pretrained(args.output_dir)

    logger.info(f"Script finished successfully, model saved in {args.output_dir}")


if __name__ == "__main__":
    main()


# https://github.com/zfjsail/gae-pytorch
# file:///C:/Users/shree/Downloads/1611.07308.pdf
def train(output_dir, wandb, glue_train_dataloader, glue_eval_dataloader, device, task, learning_rate, beta_2,
          max_train_steps=None, num_train_epochs=1, batch_size=100, logging_steps=100,
          eval_every_steps=500):
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)

    num_update_steps_per_epoch = len(glue_train_dataloader)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    else:
        num_train_epochs = math.ceil(
            max_train_steps / num_update_steps_per_epoch
        )

    num_warmup_steps = max(1000, math.floor((num_update_steps_per_epoch * 5 / 1000)))
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=learning_rate, betas=(0.9, beta_2)
    )

    def inverse_sqrt_w_warmup(step):
        if step < num_warmup_steps:
            return (num_warmup_steps - step) / num_warmup_steps

        return step ** -0.5

    lr_scheduler = LambdaLR(optimizer, lr_lambda=inverse_sqrt_w_warmup)

    logger.info("***** Running glue training *****")
    logger.info(f"  Num examples = {len(glue_train_dataloader)}")
    logger.info(f"  Num epochs = {num_train_epochs}")
    logger.info(f"  Batch size = {batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    progress_bar = tqdm(range(max_train_steps))
    global_step = 0
    model.to(device)

    # Training loop
    for epoch in range(num_train_epochs):
        model.train()  # make sure that model is in training mode, e.g. dropout is enabled
        if global_step >= max_train_steps:
            break
        # iterate over batches
        for batch in glue_train_dataloader:
            if global_step >= max_train_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            # ipdb.set_trace()

            loss = model(**batch).loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            if global_step % logging_steps == 0:
                # An extra training metric that might be useful for understanding
                # how well the model is doing on the training set.
                # Please pay attention to it during training.
                # If the metric is significantly below 80%, there is a chance of a bug somewhere.
                metric = evaluate_batch(model, batch,  task)
                wandb.log(
                    {
                        "glue_train_loss": loss,
                        "glue_learning_rate": optimizer.param_groups[0]["lr"],
                        "glue_epoch": epoch,
                        "glue_metric": metric
                    },
                    step=global_step,
                )

            if (
                    global_step % eval_every_steps == 0
                    or global_step == max_train_steps
            ):
                (
                    metric_acc
                ) = evaluate(
                    model=model,
                    eval_dataloader=glue_eval_dataloader,
                    device=device,
                    task=task
                )

                wandb.log(
                    {
                        "eval/accuracy": metric_acc,
                    },
                    step=global_step,
                )
                if global_step >= max_train_steps:
                    break

                if global_step % eval_every_steps == 0:
                    metrics = evaluate(model, glue_eval_dataloader, device, task=task)
                    wandb.log(metrics, step=global_step)

                logger.info("Saving model checkpoint to %s", output_dir)
                model.save_pretrained(output_dir)

    logger.info("Final evaluation")
    metrics = evaluate(model, glue_eval_dataloader, device, task)
    wandb.log(metrics, step=global_step)

    logger.info("Saving final model checkpoint to %s", output_dir)
    model.save_pretrained(output_dir)

    logger.info(f"Script finished successfully, model saved in {output_dir}")
