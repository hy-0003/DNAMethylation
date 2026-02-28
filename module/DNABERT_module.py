import torch
import torch.nn as nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertConfig, BertModel


'''DNABERT model'''
class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config

        # Load pre-train model
        self.kmer = config.kmer
        if self.kmer == 0:
            self.pretrainpath = f'fine_tuned_model/5hmC/DNABERT2'
            # self.pretrainpath = f'fine_tuned_model/4mC/DNABERT2'
            # self.pretrainpath = f'fine_tuned_model/6mA/DNABERT2'
            # self.pretrainpath = f'fine_tuned_model/6mA/6mATT/DNABERt2'
        else:
            self.pretrainpath = f'fine_tuned_model/5hmC/{self.kmer}/DNABERT_{self.kmer}mer'
            # self.pretrainpath = f'fine_tuned_model/4mC/{self.kmer}/DNABERT_{self.kmer}mer'
            # self.pretrainpath = f'fine_tuned_model/6mA/{self.kmer}/DNABERT_{self.kmer}mer'
            # self.pretrainpath = f'fine_tuned_model/6mA/6mATT/{self.kmer}/DNABERT_{self.kmer}mer'


        if self.kmer != 0:
            self.setting = BertConfig.from_pretrained(
                self.pretrainpath,
                num_labels=2,
                finetuning_task="dnaprom",
                cache_dir=None,
            )
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrainpath)
            self.bert = BertModel.from_pretrained(self.pretrainpath, config=self.setting)

        else:
            self.setting = BertConfig.from_pretrained(self.pretrainpath)
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrainpath, trust_remote_code=True)
            self.bert = AutoModel.from_pretrained(self.pretrainpath, trust_remote_code=True, config=self.setting)

    def forward(self, seqs):
        # DNABERT2
        if self.kmer == 0:
            if isinstance(seqs, str):
                seqs = [seqs]
            inputs = self.tokenizer(list(seqs), return_tensors='pt', padding=True, truncation=True)["input_ids"]
            if self.config.cuda:
                inputs = inputs.cuda()

            hidden_states = self.bert(inputs)[0]  # [batch_size, sequence_length, 768]
            representation = torch.mean(hidden_states, dim=1)  # [batch_size, 768]

        # DNABERT  k-mer
        else:
            seqs = list(seqs)
            kmer = [[seqs[i][x:x + self.kmer] for x in range(len(seqs[i]) + 1 - self.kmer)] for i in range(len(seqs))]
            kmers = [" ".join(kmer[i]) for i in range(len(kmer))]
            token_seq = self.tokenizer(kmers, return_tensors='pt')
            input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq[
                'attention_mask']
            if self.config.cuda:
                representation = self.bert(input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda())["pooler_output"]
            else:
                representation = self.bert(input_ids, token_type_ids, attention_mask)["pooler_output"]

        return representation


