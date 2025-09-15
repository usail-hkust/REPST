import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import GPT2Model, GPT2Config

from einops import rearrange
from reprogramming import *
from normalizer import *

class repst(nn.Module):

    def __init__(self, configs, device):
        super(repst, self).__init__()

        self.device = device
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.dropout = configs.dropout
        self.gpt_layers = configs.gpt_layers
        self.d_ff = configs.d_ff     # output mapping dimention

        self.d_model = configs.d_model
        self.n_heads= configs.n_heads
        self.d_keys = None
        self.d_llm = 768

        self.patch_nums = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        self.patch_embedding = PatchEmbedding(self.d_model, self.patch_len, self.stride, self.dropout)




        self.gpts = GPT2Model.from_pretrained('./GPT-2', output_attentions=True, output_hidden_states=True)
        self.gpts.h = self.gpts.h[:self.gpt_layers]
    
        self.gpts.apply(self.reset_parameters)

        self.word_embeddings = self.gpts.get_input_embeddings().weight.to(self.device)
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.n_vars = 5


        self.normalize_layers = Normalize(num_features=1, affine=False)
        self.mapping_layer = nn.Linear(self.vocab_size, 1)

        self.reprogramming_layer = ReprogrammingLayer(self.d_model, self.n_heads, self.d_keys, self.d_llm)
   
        self.out_mlp = nn.Sequential(
            nn.Linear(self.d_llm, 128),
            nn.ReLU(),
            nn.Linear(128, self.pred_len)
        )

        for i, (name, param) in enumerate(self.gpts.named_parameters()):
                if 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def reset_parameters(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.zeros_(module.bias) 


            
    def forward(self, x):
   

        x_enc = self.normalize_layers(x, 'norm')
     
        x_enc = rearrange(x_enc, 'b n l m -> b n m l')
        enc_out, n_vars = self.patch_embedding(x_enc)


        embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        masks = gumbel_softmax(self.mapping_layer.weight.data.permute(1,0))
        source_embeddings = self.word_embeddings[masks==1]
   
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        enc_out = self.gpts(inputs_embeds=enc_out).last_hidden_state
  
        dec_out = self.out_mlp(enc_out)

        outputs = dec_out.unsqueeze(dim=-1)      
        outputs = outputs.repeat(1, 1, 1, n_vars)

        dec_out = self.normalize_layers(outputs, 'denorm')

        return dec_out


