import math
import random
import argparse
import torch
import transformers
import logging
import wandb
import utils
import train_wnli
import LMData
import json
import time
import numpy as np
from evaluate import load
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM, 
    RobertaTokenizerFast, 
    RobertaForMaskedLM,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from datasets import load_from_disk
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from datasets import load_from_disk, DatasetDict
from tqdm.auto import tqdm
import os

class LMTrainer():
    
    def __init__(
        self,
        args,
    ):
        
        #
        self.device = args.device
        
        # step 1: get tokenizer
        self.tokenizer = self.get_tokenizer(args=args)
        
        # step 2: get dataloader
        dataloaders = self.get_dataloaders(args=args)
        self.train_dataloader = dataloaders.dataloader['train']
        self.eval_dataloader = dataloaders.dataloader['validation']
        self.test_dataloader = dataloaders.dataloader['test']
        
        # step 3: get model
        self.model = self.get_model(args=args)
        self.model_size = utils.count_parameters(self.model)
        
        # step 4: define the objective function
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion.to(self.device)
        
        # step 5: define optimizer
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), 
            lr=args.learning_rate, 
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            #amsgrad=True,
        )
        
        # step 6: define LR scheduler
        self.get_steps(args)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda=self.inverse_sqrt_w_warmup
        )
        #self.scheduler = get_cosine_schedule_with_warmup(
        #    optimizer=self.optimizer,
        #    num_warmup_steps=self.num_warmup_steps,
        #    num_training_steps=self.max_train_steps,
        #    #power=1.5,
        #)
        
        # step 6: get logger
        self.logger = logging.getLogger(__file__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        
        
        return
    
    def get_dataloaders(
        self,
        args,
    ):
        # define dataloader
        print('Reading data...')
        data_raw = load_from_disk(args.dataset_path)
        #with open(args.dataset_path, 'r') as f:
        #    data_raw = json.load(f)
        print('Reading done.')
        

        # 
        print('\nCreating dataloaders...')
        dataloaders = LMData.LMDataloader(
            dict_data=data_raw,
            tokenizer=self.tokenizer,
            mlm_probability=args.masked_percent,
            max_seq_len=args.max_seq_length,
            batch_size=args.batch_size,
            validation_size=args.validation_size,
            fixed_seed_val=args.fixed_seed_val,
            debug=args.debug,
        )
        dataloaders.check_dataloader()
    
        return dataloaders
    
    def get_tokenizer(
        self,
        args
    ):
        print('\nInitializing tokenizer...')
        # load tokenizer from the specified path
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        
        return tokenizer
    
    def get_model(
        self,
        args
    ):
        
        print('\nInitializing model...')
        #
        if not args.restart:
            # In the model, we want to follow the babyBerta configuration with our vocab size
            model = RobertaForMaskedLM.from_pretrained('phueb/BabyBERTa-3')
            config = model.config
            config.vocab_size = len(self.tokenizer) + 10 # + x is for special tokens
            model = RobertaForMaskedLM(config=config)
        else:
            model = RobertaForMaskedLM.from_pretrained(args.output_dir)
        model.to(args.device)
        
        # initialize weights of the model
        #model = self.init_weights(
        #    model=model, 
        #    fixed_seed_val=args.fixed_seed_val
        #)
        
        return model
    
    def init_weights(
        self, 
        model, 
        fixed_seed_val=0
    ):
    
        """
        Weight initialization is as follows:
        1. Non-layer norm, weight parameters = Xavier
        2. Non-layer norm, bias parameters = constant value of 0.01
        3. Layer norm, weight parameters = constant value of 1
        4. Layer norm, bias parameters = constant value of 0

        NOTE: initializations are performed with fixed seed value

        """
    
        print('\nInitializing weights of the model...')
        # fix seed
        torch.manual_seed(fixed_seed_val)
        np.random.seed(fixed_seed_val)
        random.seed(fixed_seed_val)

        #
        emb_dim = torch.tensor(model.config.hidden_size, device=model.device) 
        num_layers = torch.tensor(model.config.num_hidden_layers, device=model.device)

        for name_, par_ in model.named_parameters():
            if not (('LayerNorm' in name_) or ('layer_norm') in name_):
                if 'weight' in name_:
                    # Xavier init for weight parameters
                    torch.nn.init.xavier_normal_(par_)
                elif 'bias' in name_:
                    # const init for bias
                    torch.nn.init.constant_(par_, 0.01)
            else:
                # const init for layer norm
                if 'weight' in name_:
                    torch.nn.init.constant_(par_, 1)
                elif 'bias' in name_:
                    torch.nn.init.constant_(par_, 0)

        return model
    
    def inverse_sqrt_w_warmup(
        self,
        step,
    ):
        
        """
        NoamLR details:
        https://nn.labml.ai/optimizers/noam.html
        """
        
        #
        step = 1 if step == 0 else step
        factor = min(step**(-1 * 0.5), step * self.num_warmup_steps**(-1 * 1.5))
        return factor * (self.model_size ** (-1 * 0.5))
        
        #
        #if step < self.num_warmup_steps:
        #    return step / self.num_warmup_steps
        #else:
        #    return (step**(-0.5) * self.model_size**(-0.5))
    
    def get_steps(
        self,
        args,
    ):
        
        #
        effective_batch_size = args.batch_size * args.grad_acc_steps
        num_updates_per_epoch = math.ceil(len(self.train_dataloader) / args.grad_acc_steps)
        
        # calculate num of total_steps, total_epochs and warmup size
        if args.max_train_steps is None:
            args.max_steps = args.num_train_epochs * len(self.train_dataloader)
            args.max_train_steps = math.ceil(args.max_steps / args.grad_acc_steps)
        else:
            args.max_steps = math.ceil(args.max_train_steps * args.grad_acc_steps)
            args.num_train_epochs = math.ceil(args.max_steps / len(self.train_dataloader))
        
        # calculate warmup steps
        num_warmup_steps = min(4000, math.floor((args.max_train_steps * 5 / 100))) #max(1000, math.floor((args.max_train_steps * 5 / 100))) 
        
        # save values
        self.max_steps = args.max_steps
        self.max_train_steps = args.max_train_steps
        self.num_train_epochs = args.num_train_epochs
        self.num_warmup_steps = num_warmup_steps
        self.num_updates_per_epoch = num_updates_per_epoch
        
        #
        print(f'\nTotal forward steps, total update steps, total epochs: {self.max_steps}, {self.max_train_steps}, {self.num_train_epochs}')
        print(f'Total warmup: {self.num_warmup_steps}')
        
        return
    
    def eval_model(
        self,
        model, 
        eval_dataloader, 
        debug
    ):
        
        # turn on evlauation mode: no dropout
        #n_correct = 0
        #n_examples = 0
        #all_pred, all_trg = [], []
        total_eval_loss = torch.tensor(0.0, device=self.device)
        
        #
        model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                # to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # forward pass
                model_output = model(input_ids=input_ids, labels=labels)
                
                # loss
                loss = model_output.loss
                logits = model_output.logits
                total_eval_loss += loss
                
                
                #
                #if debug:
                #    print("predicted {}".format(torch.argmax(logits, dim=-1, keepdim=False)))
                #    print("real {}".format(labels))
                
                #
                #n_correct += (torch.argmax(logits, dim=-1, keepdim=False) == labels).sum().item()
                #n_examples += len(labels.flatten())

        # take average of loss
        eval_loss = (total_eval_loss / len(eval_dataloader)).item()
        #accuracy = n_correct / n_examples
        
        # calculate perplexity
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            logger.warning("Perplexity is infinity. Loss is probably NaN, please check your code for bugs.")
            perplexity = float("inf")

        # turn off evaluation mode
        model.train()

        return {
            "MLMEval/loss": eval_loss,
            "MLMEval/perplexity": perplexity,
            #"MLMEval/accuracy": accuracy,
        }
        
    
    def train_model(
        self,
        args
    ):
        
        # unroll objects
        train_dataloader = self.train_dataloader
        eval_dataloader = self.eval_dataloader
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        lr_scheduler = self.scheduler
        
        # progress monitors
        progress_bar = tqdm(range(self.max_steps))
        wandb.watch(model)
        
        # log
        print('\nStarting training of the model')
        logger = self.logger
        logger.info("***** Running training *****")
        logger.info(f"  Num train batches = {len(train_dataloader)}")
        logger.info(f"  Num eval batches = {len(eval_dataloader)}")
        logger.info(f"  Num epochs = {self.num_train_epochs}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        logger.info(f"  === MAIN HYPYERPAR ARE AS FOLLOWS ===")
        logger.info(f"  Batch size = {args.batch_size}")
        logger.info(f"  Gradient accumulation steps = {args.grad_acc_steps}")
        logger.info(f"  Maximum learning rate value = {args.learning_rate}")
        
        
        #
        global_step = 0
        eval_met_perp = float('inf')
        optimizer.zero_grad()
        for epoch in range(self.num_train_epochs):
            model.train()
            for batch in train_dataloader:
                
                # break if
                if global_step >= args.max_steps:
                    break
                    
                # shift tensors to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                # ipdb.set_trace()
                # print(f"input size {input_ids.shape}  label shape {labels.shape}")
                
                # forward pass
                outputs = model(input_ids=input_ids, labels=labels)
                
                # loss calculation
                # NOTE: loss = loss_, CHECKED
                loss = outputs.loss
                logits = outputs.logits.permute((0, 2, 1))
                #loss_ = criterion(logits, labels)
                #print(((loss - loss_) / loss) * 100)
                
                
                # perform gradient accumulation
                # @TODO: to average over the accumulation steps or not?
                loss = loss / args.grad_acc_steps
                loss.backward()
                if ((global_step%args.grad_acc_steps) == 0) or ((global_step+1) == args.max_train_steps):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # update progress bar
                progress_bar.update(1)
                global_step += 1
                
                
                # logging and vizualization
                if global_step % args.logging_steps == 0:
                    # An extra training metric that might be useful for understanding
                    # how well the model is doing on the training set.
                    # Please pay attention to it during training.
                    # If the metric is significantly below 80%, there is a chance of a bug somewhere
                    lr_val = optimizer.param_groups[0]["lr"]
                    
                    
                    # to wandb
                    wandb.log(
                        {
                            "MLMTraining/loss": loss,
                            "MLMTraining/learning_rate": lr_val,
                            "MLMTraining/epoch": epoch,
                        },
                        step=global_step,
                    )
                    
                    # to logger
                    #logger.info(f"Training loss (batch_loss / grad_acc_steps): {loss}")
                    #logger.info(f"Learning rate: {lr_val}")
                    
                
                # evalulate model
                if (global_step % args.eval_every_steps == 0) or ((global_step+1) == args.max_train_steps):
                    metrics = self.eval_model(model, eval_dataloader, args.debug)
                    
                    #
                    wandb.log(metrics, step=global_step)
                    
                    # to logger
                    #logger.info(f"Eval loss (average over eval set): {metrics['MLMEval/loss']}")
                    #logger.info(f"Eval perplexity: {metrics['MLMEval/perplexity']}")
                    
                    if eval_met_perp > metrics['MLMEval/perplexity']:
                        eval_met_perp = metrics['MLMEval/perplexity']
                        logger.info("Saving model checkpoint to %s", args.output_dir)
                        model.save_pretrained(args.output_dir)
                        logger.info(f"model saved in {args.output_dir}")
                
                # save model checkpoint
                if (global_step % args.save_checkpoint_evey_steps == 0):
                    model.save_pretrained(oa.path.join(args.output_dir, f'checkpoint_at_{global_step}'))
        
        # evaluate on test split
        metrics = self.eval_model(model, self.test_dataloader, args.debug)
        wandb.log(
            {
                'MLMTest/loss': metrics["MLMEval/loss"],
                'MLMTest/perplexity': metrics["MLMEval/perplexity"],
            }, 
            step=global_step
        )
                
                
        return model
