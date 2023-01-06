import math
import random
import argparse
import torch
import transformers
import logging
import wandb
import utils
#import train_wnli
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
    AlbertForMaskedLM,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from datasets import load_from_disk, load_dataset, DatasetDict, Dataset
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from datasets import load_from_disk, DatasetDict
from tqdm.auto import tqdm
import os
from copy import deepcopy
from LMModel import LanModel, LanModelConfig

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
        self.get_steps(args)
        
        # step 3: get model
        self.model = self.get_model_new(args=args)
        self.model_size = utils.count_parameters(self.model)
        args.model_size = self.model_size
        
        # step 4: define the objective function
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion.to(self.device)        
        
        # step 5: define optimizer
        self.adjusted_lr = self.get_adjusted_lr(args)
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), 
            lr=self.adjusted_lr, 
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            #amsgrad=True,
        )
        
        # step 6: define LR scheduler
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
    
    def get_adjusted_lr(
        self, 
        args
    ):
        
        # NOTE
        # This works only when we use NOAM scheduler without the model size parameter
        
        #
        target_lr = args.learning_rate
        adjusted_lr = target_lr * (self.num_warmup_steps ** 0.5)
        
        return adjusted_lr
    
    def load_wiki_data(
        self,
    ):
        
        """
        #
        wikidata = load_dataset("wikipedia", "20220301.en", beam_runner='DirectRunner')
        
        
        #
        train_test = wiki_data.train_test_split(
            test_size=0.01,
            seed=0,
        )
        test_val = train_test['test'].train_test_split(
            test_size=0.5,
            seed=0,
        )
        
        #
        data_raw = DatasetDict()
        data_raw['train'] = train_test['train']
        data_raw['validation'] = test_val['test']
        data_raw['test'] = test_val['train']
        """
        
        #
        wikidata = load_dataset("wikitext", "wikitext-103-raw-v1")#, beam_runner='DirectRunner')
        instances = []
        for split in tqdm(wikidata):
            for instance in wikidata[split]:
                if len(instance['text']) >= 50:
                    
                    if len(instance['text']) <= 110:
                        instances.append(
                            {
                                'text': instance['text']
                            }
                        )
                    else:
                        text = ''
                        for word in instance['text'].split(' '):
                            
                            if len(text) <= 100:
                                text = text + word + ' '
                            else:
                                instances.append(
                                    {
                                        'text': text
                                    }
                                )
                                text = ''
                        #
                        if text != '' and len(text) >= 30:
                            instances.append(
                                {
                                    'text': text
                                }
                            )
        
        #
        instances = Dataset.from_list(instances)
        train_test = instances.train_test_split(
            test_size=0.01,
            seed=0,
        )
        test_val = train_test['test'].train_test_split(
            test_size=0.5,
            seed=0,
        )
        
        #
        data_raw = DatasetDict()
        data_raw['train'] = train_test['train']
        data_raw['validation'] = test_val['test']
        data_raw['test'] = test_val['train']
        print(data_raw)
        
        
        return data_raw
    
    def get_dataloaders(
        self,
        args,
    ):
        # define dataloader
        print('Reading data...')
        if not args.use_wiki_data:
            data_raw = load_from_disk(args.dataset_path)
        else:
            data_raw = self.load_wiki_data()        
        print('Reading done.')
        
        #
        

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
            args=args,
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
        args,
        read_checkpoint=False,
    ):
        
        print('\nInitializing model...')
        #
        if not read_checkpoint:
            # In the model, we want to follow the babyBerta configuration with our vocab size
            model = RobertaForMaskedLM.from_pretrained('phueb/BabyBERTa-3')
            #model = AlbertForMaskedLM.from_pretrained('albert-base-v2')
            config = model.config
            config.vocab_size = len(self.tokenizer) + 10 # + x is for special tokens
            config.num_hidden_layers = args.num_hidden_layers#8
            config.num_attention_heads = args.num_attention_heads#8
            config.attention_probs_dropout_prob = 0.1
            config.hidden_dropout_prob = 0.1
            config.hidden_size = args.hidden_size#256
            config.intermediate_size = args.intermediate_size#1024
            model = RobertaForMaskedLM(config=config)
            #model = AlbertForMaskedLM(config=config)
        else:
            print(f'Reading previous checkpoint from {args.checkpoint_dir}')
            model = RobertaForMaskedLM.from_pretrained(args.checkpoint_dir)
        model.to(args.device)
        
        # initialize weights of the model
        model = self.init_weights(
            model=model, 
            fixed_seed_val=args.fixed_seed_val
        )
        
        return model
    
    def get_model_new(
        self,
        args,
        read_checkpoint=False,
    ):
        print('\nInitializing model...')
        #
        if not read_checkpoint:
            config = LanModelConfig(
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                num_attention_head=args.num_attention_heads,
                num_hidden_layers=args.num_hidden_layers,
                vocab_size=len(self.tokenizer) + 10,
            )
            model = LanModel(config=config)
        else:            
            model = AutoModel.from_pretrained(args.checkpoint_dir)
        
        #
        model.to(args.device)
        
        # initialize weights of the model
        model = self.init_weights(
            model=model, 
            fixed_seed_val=args.fixed_seed_val
        )
        
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
                if par_.dim() >= 2:
                    # Xavier init for weight parameters
                    torch.nn.init.xavier_normal_(par_)
                else:
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
        
        # Inverse SQ
        #if step < self.num_warmup_steps:
        #    return step / self.num_warmup_steps
        #else:
        #    return step**(-0.5)
        
        # NOAM
        step = 1 if step == 0 else step
        factor = min(step**(-1 * 0.5), step * self.num_warmup_steps**(-1 * 1.5))
        
        # default return value is as follows
        # factor * (self.model_size ** (-1 * 0.5))
        
        # But we want the LR schedule to be independent of the model size, so we return the following
        # factor * (1)
        
        return factor
        
        
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
        num_warmup_steps = min(4000, max(1, math.floor((args.max_train_steps * args.warmup_percent)))) #max(1000, math.floor((args.max_train_steps * 5 / 100))) 
        
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
        criterion,
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
                batch = {k_: v_.to(self.device) for k_, v_ in batch.items()}
                
                # forward pass
                outputs = model(batch['input_ids'])
                
                # loss
                loss = criterion(outputs.logits.permute(0, 2, 1), batch["labels"])
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
        
        # when to save checkpoints
        save_batch_indices = []
        for percent in range(0, 101, 10):
            save_batch_indices.append(int(np.floor(len(train_dataloader) * percent / 100)))
        
        #
        return_metrics = {
            'eval/step': [],
            'eval/batch_idx': [],
            'eval/updates': [],
            'eval/epoch': [],
            'eval/perplexity': [],
            'eval/loss': [],
            'test/perplexity': -1,
            'test/loss': -1,
        }
        
        #
        global_step = 0
        updates = 0
        eval_met_perp = float('inf')
        optimizer.zero_grad()
        for epoch in range(self.num_train_epochs):
            model.train()
            for batch_idx, batch in enumerate(train_dataloader):
                
                # break if
                if global_step >= args.max_steps:
                    break
                    
                # shift tensors to device
                batch = {k_: v_.to(self.device) for k_, v_ in batch.items()}
                
                # forward pass
                outputs = model(batch['input_ids'])
                
                # loss calculation
                # NOTE: loss = loss_, CHECKED
                #loss = outputs.loss     
                loss = criterion(outputs.logits.permute(0, 2, 1), batch["labels"])
                
                # perform gradient accumulation
                loss_acc = loss / args.grad_acc_steps
                loss_acc.backward()
                if ((global_step%args.grad_acc_steps) == 0) or ((batch_idx+1) == len(train_dataloader)):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    updates += 1
                
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
                            "MLMTraining/par_updates": updates,
                        },
                        step=global_step,
                    )
                    
                    # to logger
                    #logger.info(f"Training loss (batch_loss / grad_acc_steps): {loss}")
                    #logger.info(f"Learning rate: {lr_val}")
                    
                
                # evalulate model
                if (global_step % args.eval_every_steps == 0) or ((batch_idx+1) == len(train_dataloader)):
                    metrics = self.eval_model(
                        model=model, 
                        criterion=criterion,
                        eval_dataloader=eval_dataloader, 
                        debug=args.debug
                    )
                    
                    #
                    wandb.log(metrics, step=global_step)
                    
                    # to logger
                    #logger.info(f"Eval loss (average over eval set): {metrics['MLMEval/loss']}")
                    #logger.info(f"Eval perplexity: {metrics['MLMEval/perplexity']}")
                    
                    if eval_met_perp > metrics['MLMEval/perplexity']:
                        eval_met_perp = metrics['MLMEval/perplexity']
                        logger.info("Saving model checkpoint to %s", os.path.join(args.output_dir, "best_model"))
                        model.save_pretrained(
                            os.path.join(
                                args.output_dir,
                                "best_model",
                            )
                        )
                        best_model = deepcopy(model)
                            
                    # save values
                    return_metrics['eval/perplexity'].append(metrics['MLMEval/perplexity'])
                    return_metrics['eval/loss'].append(metrics['MLMEval/loss'])
                    return_metrics['eval/step'].append(global_step)
                    return_metrics['eval/epoch'].append(epoch)
                    return_metrics['eval/batch_idx'].append(batch_idx)
                    return_metrics['eval/updates'].append(updates)
                    
                
                # save model checkpoint
                if (global_step % args.save_checkpoint_evey_steps == 0):
                    model.save_pretrained(
                        os.path.join(
                            args.output_dir,
                            "other_checkpoints",
                            f'checkpoint_at_{global_step}'
                        )
                    )
                
                # save checkpoint with specific amount of data
                if int(batch_idx) in save_batch_indices:
                    percent_data = int((batch_idx / len(train_dataloader)) * 100)
                    model.save_pretrained(
                        os.path.join(
                            args.output_dir,
                            "data_gradient_checkpoints",
                            f'checkpoint_trained_with_{percent_data}%_data',
                        )
                    )
        
        # load the best model
        # saving a best_model instance instead of loading 
        
        # evaluate on test split
        metrics_test = self.eval_model(
            model=best_model, 
            criterion=criterion,
            eval_dataloader=self.test_dataloader, 
            debug=args.debug
        )
        
        #
        wandb.log(
            {
                'MLMTest/loss': metrics_test["MLMEval/loss"],
                'MLMTest/perplexity': metrics_test["MLMEval/perplexity"],
            }, 
            step=global_step
        )
        
        #
        return_metrics['test/perplexity'] = metrics_test["MLMEval/perplexity"]
        return_metrics['test/loss'] = metrics_test["MLMEval/loss"]
        for k_, v_ in self.model_size.items():
            return_metrics[k_] = v_
        
                
        return return_metrics
