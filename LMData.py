import os
import json
import torch
from torch.utils.data import Dataset
import transformers
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorForWholeWordMask
from tqdm import tqdm


class LMDataset(Dataset):
    def __init__(
        self,
        list_data,
        tokenizer,
        max_seq_len,
    ):
        
        #
        self.data = list_data
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
        validation_size=0.05,
        fixed_seed_val=0,
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
        - in this case we use a predefined collator, DataCollatorForWholeWordMask
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
            - validation_size: size of the validation split of the data
            - fixed_seed_val: seed value
        
        OUTPUT:
            - NONE
        
        ATTRIBUTES:
            - self.dataloader: dict
                - keys = [train, validation]
                - val = respective dataloaders
        
        ==================
        
        NOTE: 
        - use the function 'check_dataloader' to check if dataloader is correct
        
        """
        
        
        # covert dictionary of sentences into a list of grouped sentences
        list_data = self.group_sentences(
            dict_data=dict_data,
            max_len=max_seq_len,
        )
        
        # split the data
        train, val = self.split_data(
            list_data=list_data,
            validation_size=validation_size,
            fixed_seed_val=fixed_seed_val,
        )
        
        # convert into dataset format
        self.dataset = {}
        self.dataset['train'] = LMDataset(
            list_data=train,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        self.dataset['validation'] = LMDataset(
            list_data=val,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        
        # define data collator
        collator_obj = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
            return_tensors='pt',
        )
        
        #
        self.dataloader = {}
        for split in self.dataset:
            self.dataloader[split] = DataLoader(
                self.dataset[split],
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collator_obj,
            )
        
        return
        
    def group_sentences(
        self,
        dict_data,
        max_len=128,
        sentence_split_ratio=1.2,
    ):
        print('grouping sentences...')
        list_out = []
        cur_sequence = ''
        idx = 0
        for k_, v_ in tqdm(dict_data.items()):
            idx += 1
            new_sequence = cur_sequence + v_['TEXT'] + ' '
            if (len(new_sequence.split(' ')) * sentence_split_ratio) >= max_len:
                list_out.append(cur_sequence)
                cur_sequence = v_['TEXT'] + ' '
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
        np.random.seed(fixed_seed_val)
        valid_indices = np.random.choice(
            range(len(list_data)), 
            size=int(validation_size * len(list_data)),
            replace=False,
        )
        
        train_indices = list(set(list(range(len(list_data)))) - set(valid_indices))
        
        #
        #print((train_indices.__len__(), val_indices.__len__()))
        
        # save size of the data split
        self.size_train = train_indices.__len__()
        self.size_valid = valid_indices.__len__()
        
        #
        valid_, train_ = [], [] 
        idx_ = -1
        for instance_ in tqdm(list_data):
            idx_ += 1
            if idx_ in valid_indices:
                valid_.append(instance_)
            else:
                train_.append(instance_)
            
        
        #
        assert (len(valid_) + len(train_)) == len(list_data)
        assert set(valid_indices).union(train_indices) == set(list(range(len(list_data))))
        
        print('done...')
        
        return train_, valid_
    
    def check_dataloader(
        self,
        print_examples=2,
    ):
        
        #
        print(f'Size of the train dataset: {self.size_train}')
        print(f'Size of the validation dataset: {self.size_valid}')
        
        #
        for split in ['train', 'validation']:
            print(f'\nPrinting examples in the {split} split of the data')
            for batch in d_.dataloader[split]:
                check_indices = np.random.choice(
                    range(len(batch['input_ids'])), 
                    size=print_examples, 
                    replace=False
                )

                #
                for c_idx in check_indices:
                    labels = batch['labels'][c_idx]
                    in_seq = tokenizer.batch_decode(batch['input_ids'][c_idx])
                    out_seq = ''
                    masked_words = []
                    
                    #
                    for id_idx, id_ in enumerate(in_seq):
                        if id_ == '<mask>':
                            out_seq += tokenizer.decode(labels[id_idx]) + ' '
                            masked_words.append(tokenizer.decode(labels[id_idx]))
                        else:
                            out_seq += id_ + ' '

                    #
                    in_seq = ' '.join(in_seq)

                    #
                    print('='*10)
                    print(f'\nINPUT sequence is: {in_seq}')
                    print(f'\nOUTPUT sequence is: {out_seq}')
                    print(f'\nMASKED tokens: {masked_words}')
                
                #
                break
            
            #
            break
    
        return