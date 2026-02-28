import torch
import torch.nn as nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertConfig, BertModel


'''DNABERT model for CAD, Pay attention the struct does not change.'''
class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config
        self.kmer = config.kmer
        if self.kmer == 0:
            self.pretrainpath = f'fine_tuned_model/5hmC/DNABERT2'
            # self.pretrainpath = f'fine_tuned_model/4mC/DNABERT2'
            # self.pretrainpath = f'fine_tuned_model/6mA/DNABERT2'
            # self.pretrainpath = f'fine_tuned_model/6mA/6mATT/DNABERT2'
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
            self.setting = BertConfig.from_pretrained(
                self.pretrainpath,
                output_attentions=True,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrainpath,
                trust_remote_code=True
            )
            self.bert = AutoModel.from_pretrained(
                self.pretrainpath,
                trust_remote_code=True,
                config=self.setting
            )
            self.attention_outputs = []
            self._register_attention_hooks()


    def _capture_attention(self, module, input_tuple, output_tensor):
        # input_tuple[0] is attention weight, [batch_size, num_heads, seq_len, seq_len]
        if input_tuple and len(input_tuple) > 0 and isinstance(input_tuple[0], torch.Tensor):
            self.attention_outputs.append(input_tuple[0].detach().cpu())
        else:
            print("Error: Hook is used, but 'input_tuple' sample is incorrect.")

    def _register_attention_hooks(self):
        self.attention_outputs = []
        if hasattr(self.bert, 'encoder') and hasattr(self.bert.encoder, 'layer'):
            for layer in self.bert.encoder.layer:
                if hasattr(layer, 'attention') and \
                   hasattr(layer.attention, 'self') and \
                   hasattr(layer.attention.self, 'dropout'):
                    layer.attention.self.dropout.register_forward_hook(self._capture_attention)
                else:
                    print(f"Error: can not find 'attention.self.dropout' to regist hook in {layer}")
        else:
             print("Error: can not find attention leyer in self.bert.encoder.layer")

    def forward(self, seqs):
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


    def get_attention_maps(self, seqs):
        self.bert.eval()
        with torch.no_grad():
            try:
                if self.kmer == 0:
                    if isinstance(seqs, str):
                        seqs = [seqs]
                    self.attention_outputs = [] 
                    inputs_dict = self.tokenizer(list(seqs), return_tensors='pt', padding=True, truncation=True)
                    inputs = inputs_dict["input_ids"]
                    if self.config.cuda:
                        inputs = inputs.cuda()
                    _ = self.bert(inputs) 
                    if not self.attention_outputs:
                        print("="*80)
                        print("Error (kmer=0): Hooks run, but don not capture any attention")
                        print("This means the model struct had been changed or using 'attention.self' fail.")
                        print("="*80)
                        return None, None
                    attentions = tuple(self.attention_outputs) 
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs[0])


                else:
                    if isinstance(seqs, str):
                        seqs = [seqs]
                    seqs_list = list(seqs)
                    kmer = [[seqs_list[i][x:x + self.kmer] for x in range(len(seqs_list[i]) + 1 - self.kmer)] for i in range(len(seqs_list))]
                    kmers = [" ".join(kmer[i]) for i in range(len(kmer))]
                    token_seq = self.tokenizer(kmers, return_tensors='pt', padding=True, truncation=True)
                    input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq['attention_mask']
                    if self.config.cuda:
                        input_ids = input_ids.cuda()
                        token_type_ids = token_type_ids.cuda()
                        attention_mask = attention_mask.cuda()
                    outputs = self.bert(input_ids, 
                                        token_type_ids=token_type_ids, 
                                        attention_mask=attention_mask, 
                                        output_attentions=True, 
                                        return_dict=True)
                    
                    if not hasattr(outputs, 'attentions'):
                         print(f"Error (kmer={self.kmer}): your DNABERT-kmer has no '.attentions'")
                         return None, None
                    attentions = outputs.attentions
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                return attentions, tokens
            
            except Exception as e:
                print("="*80)
                print(f"Error：in 'get_attention_maps' wrong (kmer={self.kmer})")
                print(f"Error Information: {e}")
                import traceback
                traceback.print_exc()
                print("="*80)
                return None, None