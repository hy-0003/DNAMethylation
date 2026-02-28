import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients
from collections import defaultdict
import torch.cuda


class FilmTargetDimWrapper(nn.Module):
    def __init__(self, model, target_dim):
        super().__init__()
        self.model = model
        self.target_dim = target_dim
        self._device = next(model.parameters()).device

    def forward(self, berttwo_input_ids):
        berttwo_input_ids = berttwo_input_ids.to(self._device)
        attention_mask = (berttwo_input_ids != self.model.berttwo.tokenizer.pad_token_id).long()
        hidden_states = self.model.berttwo.bert(input_ids=berttwo_input_ids, attention_mask=attention_mask)[0]
        global_vector = torch.mean(hidden_states, dim=1)
        raw_sequences = []
        with torch.no_grad():
            decoded = self.model.berttwo.tokenizer.batch_decode(berttwo_input_ids, skip_special_tokens=True)
            raw_sequences = [s.replace(" ", "") for s in decoded]
        with torch.no_grad():
             kmer_vector = self.model.bertone(raw_sequences).detach()
        
        gamma, beta = self.model.film_mer(global_vector)
        modulated_mer = gamma * kmer_vector + beta
        return modulated_mer[:, self.target_dim]


def calculate_token_per_dimension_contribution_0mer(
    model, target_dim,
    sequences_pos: list, 
    sequences_neg: list,
    batch_size: int = 5
):
    device = next(model.parameters()).device
    model.to(device)
    model.eval()
    film_wrapper = FilmTargetDimWrapper(model, target_dim)
    lig = LayerIntegratedGradients(film_wrapper, model.berttwo.bert.embeddings)
    tokenizer = model.berttwo.tokenizer
    pad_token_id = tokenizer.pad_token_id
    MAX_LENGTH_0MER = 41
    token_scores_pos = defaultdict(float)
    token_counts_pos = defaultdict(int)
    for i in range(0, len(sequences_pos), batch_size):
        seq_batch = sequences_pos[i:i + batch_size]
        inputs_0mer = tokenizer(
            seq_batch, 
            return_tensors="pt", 
            padding='max_length', 
            max_length=MAX_LENGTH_0MER, 
            truncation=True
        )
        input_ids = inputs_0mer['input_ids'].to(device)
        baseline = torch.full_like(input_ids, pad_token_id) # (batch_size, seq_len)

        attributions = lig.attribute(inputs=input_ids, baselines=baseline) # (batch, seq_len, embed_dim)
        attributions_sum = attributions.sum(dim=-1).detach().cpu().numpy() # (batch, seq_len)
        
        input_ids_cpu = input_ids.cpu().numpy()
        for j in range(len(seq_batch)):
            tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu[j])
            scores_for_seq = attributions_sum[j]
            for token_str, score in zip(tokens, scores_for_seq):
                token_scores_pos[token_str] += score
                token_counts_pos[token_str] += 1
        
        del input_ids, baseline, attributions, attributions_sum, inputs_0mer
        torch.cuda.empty_cache()

    token_scores_neg = defaultdict(float)
    token_counts_neg = defaultdict(int)
    for i in range(0, len(sequences_neg), batch_size):
        seq_batch = sequences_neg[i:i + batch_size]
        
        inputs_0mer = tokenizer(
            seq_batch, 
            return_tensors="pt", 
            padding='max_length', 
            max_length=MAX_LENGTH_0MER, 
            truncation=True
        )
        input_ids = inputs_0mer['input_ids'].to(device)
        baseline = torch.full_like(input_ids, pad_token_id)

        attributions = lig.attribute(inputs=input_ids, baselines=baseline)
        attributions_sum = attributions.sum(dim=-1).detach().cpu().numpy()

        input_ids_cpu = input_ids.cpu().numpy()
        for j in range(len(seq_batch)):
            tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu[j])
            scores_for_seq = attributions_sum[j]
            for token_str, score in zip(tokens, scores_for_seq):
                token_scores_neg[token_str] += score
                token_counts_neg[token_str] += 1
        
        del input_ids, baseline, attributions, attributions_sum, inputs_0mer
        torch.cuda.empty_cache()


    all_tokens = set(token_scores_pos.keys()) | set(token_scores_neg.keys())
    delta_attributions_by_type = {}
    for token in all_tokens:
        avg_grad_pos = token_scores_pos[token] / token_counts_pos[token] if token_counts_pos[token] > 0 else 0
        avg_grad_neg = token_scores_neg[token] / token_counts_neg[token] if token_counts_neg[token] > 0 else 0
        delta_attributions_by_type[token] = avg_grad_pos - avg_grad_neg
    del film_wrapper, lig, tokenizer, pad_token_id, MAX_LENGTH_0MER, token_scores_pos, token_counts_pos, token_scores_neg, token_counts_neg, all_tokens, model
    
    return delta_attributions_by_type