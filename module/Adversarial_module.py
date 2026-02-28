import torch
# FGM(Fast Gradient Method)

class FGM():
    def __init__(self, model, emb_name1='bertone.bert.embeddings', emb_name2='berttwo.bert.embeddings'):
        self.model = model
        self.backup = {}
        self.emb_name1 = emb_name1
        self.emb_name2 = emb_name2


    def attack(self, epsilon=1.0):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (self.emb_name1 in name or self.emb_name2 in name):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 2-norm
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, ):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (self.emb_name1 in name or self.emb_name2 in name):
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
