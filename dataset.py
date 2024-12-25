import tiktoken
from tiktoken import Encoding
import torch 

#Text to Token converter 
class DataLoader:
    def __init__(self,B,T):
        self.FILE_PATH = 'input.txt'
        self.encoder : Encoding = tiktoken.get_encoding('gpt2')
        self.text, self.enc_txt = self._load_file()
        self.B, self.T     = B, T 
        self.batch_pointer = 0 
        self.total_batches = self._batches_per_epoch()
        self._stats()

    def _load_file(self):
        with open(self.FILE_PATH) as fp:
            text = fp.read()

        enc_text = self.encoder.encode(text=text)
        return text, enc_text

    def _batches_per_epoch(self):
        total_len = len(self.enc_txt)
        num_tok_per_batch =  (self.B * self.T) 
        return total_len // num_tok_per_batch

    def _stats(self):
        print(f"Total Number of Characters :{len(self.text)}")
        print(f"Total Number of Tokens     :{len(self.enc_txt)}")

    def next(self,pointer = None):
        toks = self.B * self.T 
        if pointer is  None:
            batch = self.enc_txt[(self.batch_pointer * toks)   : (self.batch_pointer + 1) * toks + 1] 
        else:
            batch = self.enc_txt[(pointer * toks)   : ( pointer + 1) * toks + 1] 

        x = torch.tensor(batch[:-1]).view(self.B, self.T)
        y = torch.tensor(batch[1 : ]).view(self.B, self.T)
        self.batch_pointer +=1 

        if self.batch_pointer * toks > len(self.enc_txt):
            self.batch_pointer = 0 
            
        return x, y 

    def reset_batch_pointer(self,):
        pass 

    def set_batch_pointer(self,):
        pass 




if __name__ == '__main__':
    data_loader = DataLoader(B=4, T = 32)
    x,y = data_loader.next()
    print(x,y)