# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Dict, Optional
from transformers import PreTrainedModel
from dataclasses import dataclass

# pair wise
class PairWiseModel(PreTrainedModel):
    def __init__(self, config, pool, model_class):
        super().__init__(config)
        self.pool = pool
        self.pretrained_model = model_class(config)
        self.fc = nn.Linear(config.d_model, 1)

    def init_pretrained_model(self, state_dict):
        self.pretrained_model.load_state_dict(state_dict) 
    
    def forward(self, input_ids, attention_mask, **kwargs):
        # (bs, seq_len) -> (bs*n_docs, seq_len)
        embeds = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.pool=='cls':
            # T5 Encoder만 따로 만들어놔야함.
            if isinstance(self.pretrained_model, T5EncoderModel):
                rpr = embeds.last_hidden_state[:,0,:] # bs, dim
            else:
                rpr = embeds['pooler_output'] # bs, dim
        elif self.pool == 'mean':
            rpr = embeds['last_hidden_state'].masked_fill(attention_mask.unsqueeze(2).repeat(1,1,self.config.hidden_size)==0,0) # bs, seq_len, dim
            rpr = rpr.sum(dim=1) # bs, dim
            s = attention_mask.sum(-1, keepdim=True) # bs, 1
            rpr = rpr/s            
            
        score = self.fc(rpr).squeeze(-1) # bs
        return dict(score = score)

class T5PairWiseModel(PreTrainedModel):
    def __init__(self, config, model_class):
        super().__init__(config)
        self.pretrained_model = model_class(config)
        self.fc = nn.Linear(config.vocab_size, 1)

    def init_pretrained_model(self, state_dict):
        self.pretrained_model.load_state_dict(state_dict) 
    
    def forward(self, input_ids, attention_mask):
        decoder_input_ids = torch.tensor([self.config.pad_token_id]*input_ids.size(0)).unsqueeze(1).to(input_ids)
        embeds = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids = decoder_input_ids)
        rpr = embeds['logits'].squeeze(1) # bs, 1, n_vocab -> bs, n_vocab
        score = self.fc(rpr).squeeze(-1) # bs
        return dict(score = score)
    
# pairwise 
class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """
    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs)
        loss = -log_probs.mean()
        return loss

# pairwise 
class MarginPairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """
    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward - margin)
        log_probs = torch.log(probs)
        loss = -log_probs.mean()
        return loss

