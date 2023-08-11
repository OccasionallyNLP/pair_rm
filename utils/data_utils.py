# -*- coding: utf-8 -*-
# data_utils
import json
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
import random
import copy
from dataclasses import dataclass
from transformers import AutoTokenizer
from itertools import combinations

DEGREE2MARGINSMALL = {1:0, 2:1/3, 3:2/3, 4:1}
DEGREE2MARGINLARGE = {1:0, 2:2, 3:1, 4:3}
DEGREE2NOMARGIN = {1:0, 2:0, 3:0, 4:0}

@dataclass
class PairWiseDataset(Dataset):
    data:List[dict]
    tokenizer:AutoTokenizer
    max_length:Optional[int]=None
    margin_type:Optional[str]='small'
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        sentence_1 = [] 
        sentence_2 = [] 
        labels = []
        degree = []
        for b in batch:
            sentence_1.append(b['sentence_1'])
            sentence_2.append(b['sentence_2'])
            if b.get('label') is not None:
                labels.append(b['label'])
            if b.get('degree') is not None:
                degree.append(b['degree'])
        if self.max_length is None:
            s1_input = self.tokenizer(sentence_1, padding='longest',return_tensors = 'pt')
            s2_input = self.tokenizer(sentence_2, padding='longest',return_tensors = 'pt')
        else:
            s1_input = self.tokenizer(sentence_1, padding=True, truncation=True, max_length=self.max_length, return_tensors = 'pt')
            s2_input = self.tokenizer(sentence_2, padding=True, truncation=True, max_length=self.max_length, return_tensors = 'pt')
        
        output = dict(sentence_1_input_ids=s1_input.input_ids, sentence_1_attention_mask=s1_input.attention_mask, 
                     sentence_2_input_ids=s2_input.input_ids,
                     sentence_2_attention_mask=s2_input.attention_mask)
        if labels:
            output['labels']=torch.tensor(labels)
        
        if degree:
            if self.margin_type=='small':
                margin = [DEGREE2MARGINSMALL[i] for i in degree]
            elif self.margin_type=='large':
                margin = [DEGREE2MARGINLARGE[i] for i in degree]
            else:
                margin = [DEGREE2NOMARGIN[i] for i in degree]
            output['margin']=torch.tensor(margin)
            output['degree']=torch.tensor(degree)
        return output

# TO DO margin
    
