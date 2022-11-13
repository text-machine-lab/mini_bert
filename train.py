import copy
import math

import numpy
from torch import rand
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BertTokenizer, RobertaForMaskedLM

import random

from functools import partial

from torch.utils.data import DataLoader
import argparse
import torch

import transformers
import datasets
from datasets import load_from_disk
import logging
from tqdm.auto import tqdm
import wandb
import utils
import train_wnli

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
        default="formatted_data",
        help="path to raw dataset",
    )

    parser.add_argument(
        "--dataset_attribute",
        type=str,
        default="qnli",
        help="glue task to evaluate on",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="path to tokenizer.  If not provided, default BERT tokenizer will be used.",
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
        "--fcn_hidden",
        default=2048,
        type=int,
        help="Hidden size of the FCN",
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
        "--eval_batch_size",
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
        default=1000,
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
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--wandb_project",
        default="mini_bert",
        help="wandb project name to log metrics to",
    )

    args = parser.parse_args()

    return args


def preprocess_function(
        examples,
        max_seq_length,
        masked_percent,
        tokenizer,
        debug,
):
    """Tokenize, truncate and add special tokens to the examples. Shift the target text by one token.

    Args:
        examples: A dictionary with keys "TEXT"
        max_seq_length: The maximum total sequence length (in tokens) for source and target texts.
        tokenizer: The tokenizer to use
        :param max_seq_length:
        :param examples:
        :param use_ast:
    """
    inputs = examples["TEXT"]
    model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=True, truncation=True, return_tensors='pt')
    model_inputs['labels'] = model_inputs.input_ids.detach().clone()

    rand_mask = torch.rand(model_inputs.labels.shape)
    # where the random array is less than 0.15, we set true
    # all tokens upto id 104 are special tokens.  id 2 is the <mask> token
    mask_arr = (rand_mask < masked_percent) * (model_inputs.input_ids > 104)
    selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
    # 2 is the masked token
    model_inputs.input_ids[0, selection] = 2

    # print(f"input size {model_inputs['input_ids'].shape}  label shape {model_inputs['labels'].shape}")

    return model_inputs


def collation_function_for_seq2seq(batch, pad_token_id):
    """
    Args:
        batch: a list of dicts of numpy arrays with keys
            input_ids
            labels
    """
    input_ids_list = [ex["input_ids"] for ex in batch]
    labels_list = [ex["labels"] for ex in batch]

    collated_batch = {
        "input_ids": utils.pad(input_ids_list, pad_token_id),
        "labels": utils.pad(labels_list, pad_token_id),
    }

    collated_batch = {
        "input_ids": utils.pad(input_ids_list, pad_token_id),
        "labels": utils.pad(labels_list, pad_token_id),
    }
    return collated_batch


def evaluate(model, eval_dataloader, device, debug):
    # turn on evlauation mode: no dropout
    model.eval()
    n_correct = 0
    n_examples = 0
    total_eval_loss = torch.tensor(0.0, device=device)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            model_output = model(input_ids=input_ids, labels=labels)
            # print("logits {}".format(logits))
            loss = model_output.loss
            logits = model_output.logits

            total_eval_loss += loss
            if debug:
                print("predicted {}".format(torch.argmax(logits, dim=-1, keepdim=False)))
                print("real {}".format(labels))
            n_correct += (torch.argmax(logits, dim=-1, keepdim=False) == labels).sum().item()
            # print("n_correct {} ".format(n_correct))
            n_examples += len(labels.flatten())

    eval_loss = (total_eval_loss / len(eval_dataloader)).item()
    accuracy = n_correct / n_examples
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        logger.warning("Perplexity is infinity. Loss is probably NaN, please check your code for bugs.")
        perplexity = float("inf")

    # turn off evaluation mode
    model.train()

    return {
        "eval/loss": eval_loss,
        "eval/perplexity": perplexity,
        "eval/accuracy": accuracy,
    }


# import ipdb


def main():
    args = parse_args()

    device = args.device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path) if args.tokenizer_path else BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # read in data
    # TODO make sure data has train and validation sets.
    # with open(args.dataset_path) as f:
    #    data_list = json.load(f)
    raw_datasets = load_from_disk(args.dataset_path)
    # print(f"dataset keys {raw_datasets.keys()}")
    if args.debug:
        raw_datasets = utils.sample_small_debug_dataset(
            raw_datasets, args.sample_size
        )
    print("Data loaded")
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    print(f"Data column_names{column_names}")
    print(f"raw dataset keys {raw_datasets.keys()}")
    preprocess_function_wrapped = partial(
        preprocess_function,
        max_seq_length=args.max_seq_length,
        masked_percent=args.masked_percent,
        tokenizer=tokenizer,
        debug=args.debug,
    )

    train_dataset = raw_datasets["train"]

    train_dataset = train_dataset.map(
        preprocess_function_wrapped,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    # print(train_dataset.column_names)

    if "validation" in raw_datasets:
        key = "validation"
    else:
        key = "test"

    _, eval_dataloader = train_wnli.prep_dataset(tokenizer, args.dataset_attribute, args.eval_batch_size,
                                                 args.sample_size, args.debug)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 2):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        logger.info(
            f"Decoded input_ids: {tokenizer.decode(train_dataset[index]['input_ids'])}"
        )
        logger.info(
            f"Decoded labels: {tokenizer.decode(train_dataset[index]['labels'], clean_up_tokenization_spaces=True)}"
        )
        logger.info("\n")

    # import ipdb; ipdb.set_trace()

    collation_function_for_seq2seq_wrapped = partial(
        collation_function_for_seq2seq,
        pad_token_id=tokenizer.pad_token_id,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collation_function_for_seq2seq_wrapped,
        batch_size=args.batch_size,
    )

    wandb.init(project=args.wandb_project, config=args)
    if args.restart:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir)
    else:
        model = RobertaForMaskedLM.from_pretrained('phueb/BabyBERTa-3')
        config = model.config
        config.vocab_size = tokenizer.vocab_size
        model = RobertaForMaskedLM(config)
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, args.beta2)
    )

    num_update_steps_per_epoch = len(train_dataloader)
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
    wandb.watch(model)
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
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
        for batch in train_dataloader:
            if global_step >= args.max_train_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            # ipdb.set_trace()
            # print(f"input size {input_ids.shape}  label shape {labels.shape}")
            loss = model(input_ids=input_ids, labels=labels).loss

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
                        "train_loss": loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            if (
                    global_step % args.eval_every_steps == 0
                    or global_step == args.max_train_steps
            ):
                (
                    metric,
                ) = train_wnli.evaluate(
                    model=model,
                    eval_dataloader=eval_dataloader,
                    device=device,
                    task=args.dataset_attribute,
                )

                wandb.log(
                    {
                        "eval metric": metric,
                    },
                    step=global_step,
                )
                if global_step >= args.max_train_steps:
                    break

                if global_step % args.eval_every_steps == 0:
                    metrics = evaluate(model=model,
                                       eval_dataloader=eval_dataloader,
                                       device=device,
                                       task=args.dataset_attribute)
                    wandb.log(metrics, step=global_step)

                logger.info("Saving model checkpoint to %s", args.output_dir)
                model.save_pretrained(args.output_dir)

    logger.info("Final evaluation")
    metrics = evaluate(model=model,
                       eval_dataloader=eval_dataloader,
                       device=device,
                       task=args.dataset_attribute)
    wandb.log(metrics, step=global_step)

    logger.info("Saving final model checkpoint to %s", args.output_dir)
    model.save_pretrained(args.output_dir)

    logger.info(f"Script finished successfully, model saved in {args.output_dir}")


if __name__ == "__main__":
    main()

# https://github.com/zfjsail/gae-pytorch
# file:///C:/Users/shree/Downloads/1611.07308.pdf
