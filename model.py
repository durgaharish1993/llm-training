import torch.nn as nn 
import torch.nn.functional as F 
from dataclasses import dataclass
import torch 
import math 
#from transformers import GPT2LMHeadModel
import time 

@dataclass
class GPTConfig:
    n_embd : int      = 768
    vocab_size : int  = 50257
    n_head : int      = 12
    block_size : int  = 1024 
    n_blocks  : int   = 12 

class MLPLayer(nn.Module):
    def __init__(self, config : GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(in_features=config.n_embd, out_features=config.n_embd *4)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(in_features=config.n_embd *4, out_features=config.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT = 1 
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x 
 
class SelfAttention(nn.Module):
    def __init__(self, config : GPTConfig):
        super().__init__()
        self.config = config
        self.c_attn  = nn.Linear(in_features=config.n_embd, out_features=config.n_embd *3)
        self.c_proj  = nn.Linear(in_features=config.n_embd, out_features=config.n_embd)
        tril_ = torch.tril(torch.ones(config.block_size, config.block_size).view(1,1,config.block_size, config.block_size))
        self.register_buffer('bias', tril_)
    def forward(self, x : torch.tensor):
        qkv    = self.c_attn(x) # (B,T, dmodel) -> (B,T, dmodel *3)
        q,k,v = qkv.split(self.config.n_embd, dim=2)  # (B,T, dmodel*3) -> [(B,T dmodel)] *3 
        (B,T,d_model) = q.size()
        h = self.config.n_head
        d = d_model//self.config.n_head
        q = q.view(B,T,h, d ).transpose(1,2)
        k = k.view(B,T,h, d ).transpose(1,2)
        v = v.view(B,T,h, d ).transpose(1,2)
        att = q @ k.transpose(-1,-2) * (1/math.sqrt(d))  # (B,h,T,d) @ (B,h,d,T) -> (B,h,T,T)
        att = att.masked_fill(self.bias[:,:,:T, :T]== 0, float('-inf'))
        out = F.softmax(att, dim=-1) @ v   # (B, h, T,T) @ (B,h, T, d)-> (B,h, T, d)
        out = out.transpose(1,2).contiguous().view(B,T,d_model) #(B,h, T,  d) -> (B, T, d_model)
        out = self.c_proj(out)
        return out 
    
class AttentionWrapper(nn.Module):
    def __init__(self,config : GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config=config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLPLayer(config)

    def forward(self, x : torch.Tensor):
        x = x + self.attn(self.ln_1(x)) 
        x = x + self.mlp(self.ln_2(x))

        return x 


class GPTModel(nn.Module):
    def __init__(self,config : GPTConfig):
        super().__init__()
        self.transformer = nn.ModuleDict(modules={
            "wte" : nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embd),
            "wpe" : nn.Embedding(num_embeddings=config.block_size, embedding_dim=config.n_embd),
            "h"   : nn.ModuleList([ AttentionWrapper(config=config) for _ in range(config.n_blocks)]),
            "ln_f" : nn.LayerNorm(config.n_embd)
         })
        self.config = config 
        
        self.lm_head = nn.Linear(in_features=config.n_embd, out_features=config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight 
        # Reference Papers - Using the Output Embedding to Improve Language Models (https://arxiv.org/pdf/1608.05859)
        # GPT2, Language Models are Unsupervised Multitask Learners (https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

        self.apply(self._initialize_weights)

        # Residual Networks (https://arxiv.org/pdf/1512.03385)

    

    def _initialize_weights(self, module : nn.Module):
        # Implementation of weights from Original GPT2 architecutre 
        # (https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L152)

        if isinstance(module, nn.Linear):
            std = 0.02 
            if hasattr(module, 'NANO_GPT_SCALE_INIT'):
                std *= (2 * self.config.block_size) ** -0.5  # 2 comes from the two residual paths 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if isinstance(module, nn.LayerNorm):
            # Gamma (weight): Initialized to 1 using init.ones_().
            # Beta (bias): Initialized to 0 using init.zeros_().
            torch.nn.init.ones_(module.weight) # Scale 
            torch.nn.init.zeros_(module.bias)  # Offset 
            #No need to inititailize this - As defaults are 1, 0. 


    def forward(self, in_tok, targets=None ):
        (B,T) = in_tok.size()
        pos = torch.arange(0, T, dtype=torch.long, device=in_tok.device)
        tok_emb   = self.transformer.wte(in_tok) 
        pos_emb   = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
        count = 0 
        out_dict = {}
        for block in self.transformer.h : 
            x = block(x)
            out_dict[count] = x
            count += 1  

        x  = self.transformer.ln_f(x)
        logits  = self.lm_head(x) # (B, T, vocab_size)
        loss = None 
        if targets is not None:
            loss = F.cross_entropy(input= logits.view(-1,logits.size(-1)), target= targets.view(-1) )


        RETURN_DETAILED = False 
        if RETURN_DETAILED:
            return_obj = logits, loss, tok_emb, pos_emb, out_dict
        else:
            return_obj = logits, loss 
        return return_obj
    

    @classmethod
    def load_pretrained_weights(cls):

        gpt_config = GPTConfig()
        model = GPTModel(config=gpt_config)

        gpt_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        pre_trained_weights  = gpt_hf.state_dict()

        state_dict_torch = model.state_dict()
        py_keys = [key for key in state_dict_torch.keys() if not key.endswith('.attn.bias')]
        hf_keys = [ key for key in pre_trained_weights.keys()  if not key.endswith('.attn.masked_bias')  ]

        transpose_tensors = ['.attn.c_attn.weight','.attn.c_attn.weight', '.mlp.c_fc.weight', '.mlp.c_proj.weight', '.attn.c_proj.weight']
        print("copying the data")
        for key in hf_keys:
            list_check = [key.endswith(t_tensor) for t_tensor in transpose_tensors]
            if True in list_check:
                assert pre_trained_weights[key].shape[::-1] == state_dict_torch[key].shape

                with torch.no_grad():
                    state_dict_torch[key].copy_(pre_trained_weights[key].t())
            
            else:
                assert pre_trained_weights[key].shape == state_dict_torch[key].shape
                with torch.no_grad():
                    state_dict_torch[key].copy_(pre_trained_weights[key])


        return model, pre_trained_weights




from dataset import DataLoader

@dataclass 
class TrainConig:
    pass 

class Trainer:
    def __init__(self, data_loader, model, device = 'cpu' ):
        self.data_loader : DataLoader  = data_loader
        self.model       : nn.Module   = model 
        self.optimer     = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.device      = device 
    def train(self):
        
        print("in training loop")
        for i in range(50):
            s_time = time.time()
            x,y = self.data_loader.next()
            x = x.to(device)
            y = y.to(device)
            self.optimer.zero_grad()
            (B,T) =  x.size()
            logits, loss = self.model(x,y)
            #import code ; code.interact(local=locals())
            loss.backward()
            self.optimer.step()
            e_time = time.time()
            time_diff = (e_time - s_time) * 1000
            tok_sec   =  (B * T)/ (e_time - s_time)
            print(f"step {i}, loss : {loss.item()}, time : {time_diff : .2f}ms, tok/sec : {tok_sec : .1f} ")

        
            
if __name__ == "__main__":

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'gpu'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Device Used for running the code :: {device}")

    torch.manual_seed(1337)
    gpt_conf    = GPTConfig()
    gpt_model   = GPTModel(config=gpt_conf)
    data_loader = DataLoader(B=4, T = 32)
    gpt_model.to(device)
    train_obj = Trainer(data_loader=data_loader, model=gpt_model, device=device)
    print("training loop")
    train_obj.train()








