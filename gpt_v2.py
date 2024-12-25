import torch.nn as nn
from dataclasses import dataclass
import torch 
import torch.nn.functional as F 
import math 
class GPT2Conf:
    vocab_size    : int = 50257
    n_embed       : int = 768
    block_size    : int = 1024 
    n_blocks      : int = 12 
    n_heads       : int = 12


class GPTMLP(nn.Module):
    def __init__(self,config : GPT2Conf):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embed, config.n_embed * 4 )
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embed *4 , config.n_embed)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

         

class GPTAttention(nn.Module):
    def __init__(self,config : GPT2Conf):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embed, config.n_embed *3) 
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.config = config 
        tril_ = torch.tril(torch.ones(config.block_size, config.block_size).view(1,1,config.block_size, config.block_size))
        self.register_buffer('bias', tril_)


    def forward(self,x):
        #(B,T,n_embed) -> (B,T, n_embed *3)
        qkv = self.c_attn(x)
        #(B,T, n_embed *3) -> 3 * (B,T, n_embed)
        q,k,v = qkv.split(self.config.n_embed, 2)
        
        (B,T,n_embed) = q.size()
        h = self.config.n_heads
        d = n_embed//h

        q = q.view(B,T, h, n_embed//h ).transpose(1,2) # (B, h, T, d)
        k = k.view(B,T, h, n_embed//h ).transpose(1,2) # (B, h, T, d)
        v = v.view(B,T, h, n_embed//h ).transpose(1,2) # (B, h, T, d)

        #(B,h,T,d) x (B,h,d,®T) = (T,d) x (d,T) on index (B,h) => (T,T) on index (B,h)
        # (j>i) apply mask
        attn_weights = q @ k.transpose(-1,-2)
        attn_weights = attn_weights.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
       
        attn = F.softmax(attn_weights/ math.sqrt(d),dim=-1)
        attn = attn @ v 
        # (B,h, T,T) x (B,h, T, d) -> (B,h,T,d) 
        # (T,T) x (T,d) 
        out = attn.transpose(1,2).contiguous().view(B,T, h*d)
        out = self.c_proj(out)
        return out 









         


    

class GPTBlock(nn.Module):
    def __init__(self, config : GPT2Conf):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embed) 
        self.attn = GPTAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)  
        self.mlp  = GPTMLP(config )
    
    def forward(self,x):
        #(B,T,n_embed) -> (B,T,n_embed) -> 
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))



class GPT2Model(nn.Module):
    def __init__(self, config : GPT2Conf):
        super().__init__()
        self.config = config 

        self.transformer = nn.ModuleDict({
                        "wte"  : nn.Embedding(config.vocab_size, config.n_embed),
                        "wpe"  : nn.Embedding(config.block_size, config.n_embed), 
                        "h"    : nn.ModuleList([ GPTBlock(config=config)  for i in range(config.n_blocks) ]) ,
                        "ln_f" : nn.LayerNorm(config.n_embed)  

                            })
        
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    
    def forward(self, toks ):
        # (B,T) -> (B,T,n_embed)
        (B,T) = toks.size()
        tok_emb = self.transformer.wte(toks)
        pos = torch.arange(0,T).view(1,T)
        # (1,T) -> (1,T, n_embed)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        #(B,T,n_embed) -> (B,T,n_embed)
        for block in self.transformer.h:
            x = block(x)

        #(B,T,n_embed) -> (B,T,n_embed)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits 

        


        


        




