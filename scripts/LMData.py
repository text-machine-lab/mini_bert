import os
import json
import torch
from torch.utils.data import Dataset
import transformers
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorForWholeWordMask, DataCollatorForLanguageModeling
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys

class LMDataset(Dataset):
    def __init__(
        self,
        list_data,
        tokenizer,
        max_seq_len,
        debug=False,
    ):
        
        #
        try:
            self.data = list_data['TEXT']
        except:
            self.data = list_data['text']
        if debug:
            self.data = self.data[:1000]
        
        #
        self.tokenizer = tokenizer
        self.max_len = max_seq_len
    
        return
    
    def __len__(
        self,    
    ):
        return len(self.data)
    
    def __getitem__(
        self,
        idx,
    ):
        #
        example = self.data[idx]
        
        # tokenize
        tokenized = self.tokenizer.encode_plus(
            text=example,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
        )
        
        #
        #tokenized['labels'] = tokenized['input_ids']
        
        return tokenized

class LMDataloader():
    
    def __init__(
        self, 
        dict_data,
        tokenizer,
        mlm_probability,
        max_seq_len,
        batch_size=8,
        fixed_seed_val=0,
        debug=False,
        args=None,
    ):
        
        """
        This class takes raw data as input and performs following operations,
        
        1. sentence grouping: 
        - our raw dataset is comprised of sentences filtered based on AOChildes vocabulary 
        from multiple sources
        - if we use dataset as is, we will only train our model on ~15 tokens. We need to 
        train the model on longer input sequences. Hence, we need to group the sentences.
        - Currently, we only have BOS abd EOS tokens separating various sentences
        
        2. split the data into train and validation splits
        - after grouping the sentences, total data instances will reduce. We split the reduced
        set into two sets, train and validation.
        
        3. convert splitted data into the required 'dataset' object
        - to make it compatible with huggingface 
        
        4. define data collator
        - in this case we use a predefined collator, DataCollatorForLanguageModeling
        - this data collator only needs 'input_ids' and the 'input_ids' it will create
        'labels' i.e. target sequence for each input sequence
        - 'labels' contain masked token ids and '-100's. '-100' values are there to tell
        objective function not to calculate loss at those positions.
        - In other words, we only calculate loss corresponding to the masked tokens
        
        5. create dataloader
        - create dataloader for each data split
        
        ==================
        
        INPUT: 
            - dict_data: this is raw data read from the .json file
            - tokenizer: pre-trained tokenizer
            - mlm_probability: how much to mask?
            - max_seq_len: maximum sequence length?
            - batch_size: batch size in the dataloader
            - fixed_seed_val: seed value
            - debug: only takes 10 data instances if true
        
        OUTPUT:
            - NONE
        
        ATTRIBUTES:
            - self.dataloader: dict
                - keys = [train, validation]
                - val = respective dataloaders
        
        ==================
        
        NOTE: 
        - use the function 'check_dataloader' to check if dataloader is correct
        
        - Default collation object (defined above) matches the masking adopted in BERT
        i.e., 
        15% masking, 
        80% of the words selected for masking are replaced by  <mask>
        10% of the words selected for masking are kept the same
        10% of the words selected for masking are replaced by a random word
        And, we do not have control over (80, 10, 10).
        
        The custom collation function defined in this class simplifies the masking 
        and implements 15% masking with (100, 0, 0).
        
        
        """
        
        #
        self.debug = debug
        self.tokenizer = tokenizer
        self.fixed_seed_val = fixed_seed_val
        self.max_seq_len = max_seq_len
        self.mlm_probability = mlm_probability
        
        # convert into dataset format
        self.dataset = {}
        for split in ['train', 'validation', 'test']:
            self.dataset[split] = LMDataset(
                list_data=dict_data[split],
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                debug=debug,
            )

        #
        self.size_train = self.dataset['train'].__len__()
        self.size_valid = self.dataset['validation'].__len__()
        
        # define data collator
        collator_obj = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
            return_tensors='pt',
        )
        
        #
        self.dataloader = {}
        for split in ['train', 'validation', 'test']:#self.dataset:
            if (not split == 'train') and (not args == None):
                bs_ = args.eval_batch_size
            else:
                bs_ = batch_size
            
            #
            self.dataloader[split] = DataLoader(
                self.dataset[split],
                batch_size=bs_,
                shuffle=False,
                collate_fn=self.custom_collation, #collator_obj, #self.custom_collation
            )
        
        return
        
    def group_sentences(
        self,
        dict_data,
        max_len=128,
        sentence_split_ratio=1.2,
        key='TEXT'
    ):
        print('grouping sentences...')
        list_out = []
        cur_sequence = ''
        idx = 0
        for k_, v_ in tqdm(dict_data.items()):
            idx += 1
            new_sequence = cur_sequence + v_[key] + ' '
            if (len(new_sequence.split(' ')) * sentence_split_ratio) >= max_len:
                list_out.append(cur_sequence)
                cur_sequence = v_[key] + ' '
            else:
                cur_sequence = new_sequence
            
            #
            #if idx > 10000:
            #    break
        print('done')
        
        return list_out
    
    def split_data(
        self,
        list_data,
        validation_size,
        fixed_seed_val,
    ):
        
        #
        print('splitting data...')
        
        #
        if self.debug:
            list_data = list_data#[:10]
        
        #
        train_, valid_ = train_test_split(
            list_data,
            test_size=validation_size,
            random_state=1,#fixed_seed_val
        )
        
        #
        self.size_train = train_.__len__()
        self.size_valid = valid_.__len__()
        
        #
        #assert (len(valid_) + len(train_)) == len(list_data)
        
        print('done...')
        
        return train_, valid_
    
    def custom_collation(
        self,
        batch,
    ):
        
        #
        input_ids = [i['input_ids'] for i in batch]
        np.random.seed()
        
        seq_ins = []
        seq_outs = []
        for seq in input_ids:
            seq_in = [self.tokenizer.pad_token_id] * self.max_seq_len
            seq_in[:len(seq)] = seq
            
            #
            mask_pos = (np.random.uniform(low=0, high=1, size=len(seq_in)) <= self.mlm_probability).tolist()
            seq_out = [self.tokenizer.pad_token_id] * self.max_seq_len
            for pos_idx, pos in enumerate(seq_in):
                if pos == self.tokenizer.pad_token_id:
                    break
                
                if mask_pos[pos_idx] == 1:
                    seq_out[pos_idx] = seq[pos_idx]
                    seq_in[pos_idx] = self.tokenizer.mask_token_id
                else:
                    seq_out[pos_idx] = -100
            
            #
            seq_ins.append(seq_in)
            seq_outs.append(seq_out)
        
        #
        batch_out = {
            'input_ids': torch.LongTensor(seq_ins),
            'labels': torch.LongTensor(seq_outs)
        }
        
        #
        np.random.seed(self.fixed_seed_val)   
            
        return batch_out
    
    def check_dataloader(
        self,
        print_examples=2,
    ):
        
        #
        print(f'Size of the train dataset: {self.size_train}')
        print(f'Size of the validation dataset: {self.size_valid}')
        
        #
        count_add_pred = 0
        for split in ['train', 'validation']:
            print(f'\nPrinting examples in the {split} split of the data')
            for batch in self.dataloader[split]:
                check_indices = np.random.choice(
                    range(len(batch['input_ids'])), 
                    size=print_examples, 
                    replace=False
                )

                #
                for c_idx in check_indices:
                    labels = batch['labels'][c_idx]
                    in_seq = self.tokenizer.batch_decode(batch['input_ids'][c_idx])
                    out_seq = ''
                    masked_words = []
                    predictions = []
                    prediction_ids = []
                    
                    #
                    count_mask = 0
                    for id_idx, id_ in enumerate(in_seq):
                        if id_ == '<mask>':
                            count_mask += 1
                            out_seq += self.tokenizer.decode(labels[id_idx]) + ' '
                            masked_words.append(self.tokenizer.decode(labels[id_idx]))
                        else:
                            out_seq += id_ + ' '
                    
                    #
                    for id_idx, id_ in enumerate(labels):
                        if id_ != -100:
                            predictions.append(self.tokenizer.decode(id_))
                            prediction_ids.append(id_)
                    #
                    in_seq = ' '.join(in_seq)
                    
                    #
                    print('='*10)
                    print(f'\nINPUT sequence is: {in_seq}')
                    print(f'\nOUTPUT sequence is: {out_seq}')
                    print(f'\nMASKED tokens: {masked_words}')

                    #
                    if count_mask != len(predictions):
                        count_add_pred += 1                 
                        #print('='*10)
                        #print(f'\nPRED: {predictions}')
                        #print(f'\nPRED IDs: {prediction_ids}')
                #
                break
            
            #
            break
        #print(f'Count of examples with added prediction: {count_add_pred}')
        
        return