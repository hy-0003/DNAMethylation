import torch
import torch.nn as nn
import torch.nn.functional as F
from module import DNABERT_module


'''Fusion Model'''
class FiLM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, output_dim*2)
        )
        
    def forward(self, global_vector):
        params = self.mlp(global_vector)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return gamma, beta


class MLPExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

    def forward(self, x):
        return self.layer(x)

class TransformerExpert(nn.Module):
    def __init__(self, input_dim, nhead=4, dim_feedforward=256, dropout=0.5, num_chunks=4):
        super().__init__()
        
        assert input_dim % num_chunks == 0, "input_dim must be divisible by num_chunks"
        self.num_chunks = num_chunks
        self.chunk_dim = input_dim // num_chunks
        assert self.chunk_dim % nhead == 0, "chunk_dim must be divisible by nhead"

        # Define Transformer Encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.chunk_dim,            
            nhead=nhead,                       
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True                 
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)
        self.output_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x_seq = x.view(batch_size, self.num_chunks, self.chunk_dim)
        transformer_output = self.transformer_encoder(x_seq)
        output_flat = transformer_output.view(batch_size, -1)
        output = self.output_norm(x + output_flat)
        return output


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        weights = F.softmax(self.gate(x), dim=-1)
        return weights


class Fusion(nn.Module):
    """FiLM→MoE Fusion model"""        
    def __init__(self, config,
                 kmer_dim=768,         
                 global_dim=768,       
                 num_experts=4,        
                 # Transformer
                 expert_nhead=4,            
                 expert_ff_dim=256,        
                 expert_num_chunks=4
                ):
        super().__init__()
        self.config = config
        self.config.kmer = self.config.kmers[0]
        self.bertone = DNABERT_module.BERT(self.config)
        self.config.kmer = self.config.kmers[1]
        self.berttwo = DNABERT_module.BERT(self.config)

        self.classification = nn.Sequential(
            nn.Linear(768, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

        self.film_mer = FiLM(global_dim, kmer_dim)
        self.experts = nn.ModuleList([
            TransformerExpert(
                input_dim=kmer_dim,
                nhead=expert_nhead,
                dim_feedforward=expert_ff_dim,
                dropout=0.5,
                num_chunks=expert_num_chunks
            ) for _ in range(num_experts)
        ])
        self.gating = GatingNetwork(global_dim, num_experts)
        self.output_layer = nn.Linear(kmer_dim, kmer_dim)
        
    def forward(self, seqs):
        kmer = self.bertone(seqs)
        global_vector = self.berttwo(seqs)

        gamma, beta = self.film_mer(global_vector)
        modulated_mer = gamma * kmer + beta

        expert_weights = self.gating(global_vector)
        # Expert to k-mer
        expert_outputs = []
        for expert in self.experts:
            output = expert(modulated_mer)
            expert_outputs.append(output)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, kmer_dim]
        weighted_output = torch.sum(expert_outputs * expert_weights.unsqueeze(-1), dim=1)

        final_output = self.output_layer(weighted_output)
        output = self.classification(final_output)
        return output, final_output