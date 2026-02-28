import torch
from captum.attr import IntegratedGradients
from collections import defaultdict
import torch.cuda

def dna_to_kmers(seq, k=6):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, target_dim):
        super().__init__()
        self.model = model
        self.target_dim = target_dim

    def forward(self, kmer_vector, global_vector):
        gamma, beta = self.model.film_mer(global_vector)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        modulated_mer = gamma * kmer_vector + beta
        return torch.sum(modulated_mer[:, :, self.target_dim], dim=1)


def calculate_token_per_dimension_contribution_6mer(
    model, target_dim,
    sequences_pos: list,
    sequences_neg: list,
    batch_size: int = 5
):
    device = next(model.parameters()).device
    model.to(device)
    model.eval()
    ig = IntegratedGradients(ModelWrapper(model, target_dim))
    MAX_LENGTH_0MER = 41
    kmer_scores_pos = defaultdict(float)
    kmer_counts_pos = defaultdict(int)
    
    for i in range(0, len(sequences_pos), batch_size):
        seq_batch = sequences_pos[i:i + batch_size]
        with torch.no_grad():
            kmer_inputs = [model.bertone.tokenizer.encode(s) for s in seq_batch]
            kmer_embeddings_batch = model.bertone.bert.embeddings(torch.tensor(kmer_inputs, device=device)).detach()
            inputs_0mer = model.berttwo.tokenizer(
                seq_batch, 
                return_tensors="pt", 
                padding='max_length', 
                max_length=MAX_LENGTH_0MER, 
                truncation=True
            ).to(device)
            attention_mask = (inputs_0mer['input_ids'] != model.berttwo.tokenizer.pad_token_id).long()
            hidden_states = model.berttwo.bert(input_ids=inputs_0mer['input_ids'], attention_mask=attention_mask)[0]
            global_vectors_batch = torch.mean(hidden_states, dim=1).detach()

        kmer_embeddings_batch.requires_grad_()
        baseline = torch.zeros_like(kmer_embeddings_batch)
        attributions = ig.attribute(
            inputs=kmer_embeddings_batch, 
            baselines=baseline, 
            additional_forward_args=(global_vectors_batch,)
        )
        attributions_sum = attributions.sum(dim=2).detach().cpu().numpy() # (batch_size, seq_len)
        for j in range(len(seq_batch)):
            kmers_in_seq = dna_to_kmers(seq_batch[j], k=6)
            scores_for_seq = attributions_sum[j][:len(kmers_in_seq)] 
            
            for kmer_str, score in zip(kmers_in_seq, scores_for_seq):
                kmer_scores_pos[kmer_str] += score
                kmer_counts_pos[kmer_str] += 1
        
        del kmer_embeddings_batch, global_vectors_batch, baseline, attributions, attributions_sum, inputs_0mer
        torch.cuda.empty_cache()


    kmer_scores_neg = defaultdict(float)
    kmer_counts_neg = defaultdict(int)
    for i in range(0, len(sequences_neg), batch_size):
        seq_batch = sequences_neg[i:i + batch_size]
        with torch.no_grad():
            kmer_inputs = [model.bertone.tokenizer.encode(s) for s in seq_batch]
            kmer_embeddings_batch = model.bertone.bert.embeddings(torch.tensor(kmer_inputs, device=device)).detach()
            inputs_0mer = model.berttwo.tokenizer(
                seq_batch, 
                return_tensors="pt", 
                padding='max_length', 
                max_length=MAX_LENGTH_0MER, 
                truncation=True
            ).to(device)
            attention_mask = (inputs_0mer['input_ids'] != model.berttwo.tokenizer.pad_token_id).long()
            hidden_states = model.berttwo.bert(input_ids=inputs_0mer['input_ids'], attention_mask=attention_mask)[0]
            global_vectors_batch = torch.mean(hidden_states, dim=1).detach()

        kmer_embeddings_batch.requires_grad_()
        baseline = torch.zeros_like(kmer_embeddings_batch)
        attributions = ig.attribute(
            inputs=kmer_embeddings_batch, 
            baselines=baseline, 
            additional_forward_args=(global_vectors_batch,)
        )
        attributions_sum = attributions.sum(dim=2).detach().cpu().numpy()

        for j in range(len(seq_batch)):
            kmers_in_seq = dna_to_kmers(seq_batch[j], k=6)
            scores_for_seq = attributions_sum[j][:len(kmers_in_seq)]
            
            for kmer_str, score in zip(kmers_in_seq, scores_for_seq):
                kmer_scores_neg[kmer_str] += score
                kmer_counts_neg[kmer_str] += 1
        
        del kmer_embeddings_batch, global_vectors_batch, baseline, attributions, attributions_sum, inputs_0mer
        torch.cuda.empty_cache()

    all_kmers = set(kmer_scores_pos.keys()) | set(kmer_scores_neg.keys())
    delta_attributions_by_type = {}
    for kmer in all_kmers:
        avg_grad_pos = kmer_scores_pos[kmer] / kmer_counts_pos[kmer] if kmer_counts_pos[kmer] > 0 else 0
        avg_grad_neg = kmer_scores_neg[kmer] / kmer_counts_neg[kmer] if kmer_counts_neg[kmer] > 0 else 0
        delta_attributions_by_type[kmer] = avg_grad_pos - avg_grad_neg
    del ig, MAX_LENGTH_0MER, model, kmer_scores_pos, kmer_counts_pos, kmer_scores_neg, kmer_counts_neg, all_kmers
    
    return delta_attributions_by_type