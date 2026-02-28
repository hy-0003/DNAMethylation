import torch, os, sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
try:
    from configuration import config_init
    from module.Fusion_module import Fusion
    from util import util_file
except ImportError:
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(rootPath)
    from configuration import config_init
    from module.Fusion_module import Fusion
    from util import util_file


class FusionAnalyzer:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        self.handles = []
        self.storage = {}

    def _register_hooks(self):
        def film_hook(module, input, output):
            gamma, beta = output
            self.storage["film_gamma"] = gamma.detach()
            self.storage["film_beta"] = beta.detach()
        self.handles.append(self.model.film_mer.register_forward_hook(film_hook))
        def gating_hook(module, input, output):
            self.storage["gating_weights"] = output.detach()
        self.handles.append(self.model.gating.register_forward_hook(gating_hook))
        for i, expert in enumerate(self.model.experts):
            def make_hook(idx):
                def expert_hook(module, input, output):
                    self.storage[f"expert_{idx}"] = output.detach()
                return expert_hook
            self.handles.append(expert.register_forward_hook(make_hook(i)))

    def analyze(self, seqs: list):
        self.storage = {}
        self._register_hooks()
        self.model.eval()
        with torch.no_grad():
            _ = self.model(seqs)
        for h in self.handles:
            h.remove()
        self.handles = []
        return self.storage
    

def FILM_Impact_Analysis(config, model_path, data_dir, output_dir):
    P_THRESHOLD = 0.05
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    model = Fusion(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    data_path = data_dir
    sequences, labels = util_file.load_tsv_format_data(data_path)
    pos_kmer_vecs, pos_gammas, pos_betas = [], [], []
    neg_kmer_vecs, neg_gammas, neg_betas = [], [], []
    analyzer = FusionAnalyzer(model)
    batch_size = 2048
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            seq_batch = sequences[i:i + batch_size]
            label_batch = np.array(labels[i:i + batch_size])
            storage = analyzer.analyze(seq_batch)
            gammas = storage["film_gamma"].cpu().numpy()
            betas = storage["film_beta"].cpu().numpy()
            kmer_vecs = model.bertone(seq_batch).cpu().numpy()
            pos_mask = (label_batch == 1)
            neg_mask = (label_batch == 0)

            if pos_mask.any():
                pos_kmer_vecs.append(kmer_vecs[pos_mask])
                pos_gammas.append(gammas[pos_mask])
                pos_betas.append(betas[pos_mask])

            if neg_mask.any():
                neg_kmer_vecs.append(kmer_vecs[neg_mask])
                neg_gammas.append(gammas[neg_mask])
                neg_betas.append(betas[neg_mask])

    pos_kmer_vecs, pos_gammas, pos_betas = np.concatenate(pos_kmer_vecs), np.concatenate(pos_gammas), np.concatenate(pos_betas)
    neg_kmer_vecs, neg_gammas, neg_betas = np.concatenate(neg_kmer_vecs), np.concatenate(neg_gammas), np.concatenate(neg_betas)
    n_pos = len(pos_kmer_vecs)
    n_neg = len(neg_kmer_vecs)
    pos_impact_vectors = (pos_gammas * pos_kmer_vecs + pos_betas) - pos_kmer_vecs
    neg_impact_vectors = (neg_gammas * neg_kmer_vecs + neg_betas) - neg_kmer_vecs
    num_dims = pos_impact_vectors.shape[1]
    avg_pos_impact = np.mean(pos_impact_vectors, axis=0)
    avg_neg_impact = np.mean(neg_impact_vectors, axis=0)
    pull_apart_contribution = avg_pos_impact - avg_neg_impact
    std_pos_all_dims = np.std(pos_impact_vectors, axis=0, ddof=1)
    std_neg_all_dims = np.std(neg_impact_vectors, axis=0, ddof=1)
    s_pooled_all_dims = np.sqrt(((n_pos - 1) * std_pos_all_dims**2 + (n_neg - 1) * std_neg_all_dims**2) / (n_pos + n_neg - 2))
    cohen_d_all_dims = np.divide(pull_apart_contribution, s_pooled_all_dims, 
                                 out=np.zeros_like(pull_apart_contribution), 
                                 where=s_pooled_all_dims!=0)

    impact_p_values = []
    for i in range(num_dims):
        p_val = ttest_ind(
            pos_impact_vectors[:, i], 
            neg_impact_vectors[:, i], 
            equal_var=False
        ).pvalue
        impact_p_values.append(p_val)


    df_data = {
        "维度": np.arange(num_dims),
        "正样本平均变化量": avg_pos_impact,
        "负样本平均变化量": avg_neg_impact,
        "拉远贡献值": pull_apart_contribution,
        "科恩d值": cohen_d_all_dims,  # <--- 新增列
        "拉远贡献的P值": impact_p_values
    }
    df_results = pd.DataFrame(df_data)
    df_results['abs_cohen_d'] = df_results['科恩d值'].abs() # <--- 使用科恩d值
    df_credible = df_results[df_results['拉远贡献的P值'] <= P_THRESHOLD]
    df_less_credible = df_results[df_results['拉远贡献的P值'] > P_THRESHOLD]
    print(f"  找到 {len(df_credible)} 个显著维度 (P <= {P_THRESHOLD})")
    print(f"  找到 {len(df_less_credible)} 个不显著维度 (P > {P_THRESHOLD})")
    df_credible_sorted = df_credible.sort_values(by='abs_cohen_d', ascending=False)
    df_less_credible_sorted = df_less_credible.sort_values(by='拉远贡献的P值', ascending=True)
    df_final_sorted = pd.concat([df_credible_sorted, df_less_credible_sorted])
    df_final_sorted = df_final_sorted.drop(columns=['abs_cohen_d'])
    csv_path = os.path.join(output_dir, "Impact_Analysis.csv")
    df_final_sorted.to_csv(csv_path, index=False, float_format='%.8e')


# 5hmC   '5hmC_H.sapiens','5hmC_M.musculus'
# 4mC    '4mC_C.equisetifolia','4mC_F.vesca','4mC_S.cerevisiae','4mC_Tolypocladium'
# 6mA    '6mA_A.thaliana''6mA_C.elegans''6mA_C.equisetifolia''6mA_D.melanogaster'
# 6mA    '6mA_F.vesca''6mA_H.sapiens''6mA_R.chinensis''6mA_S.cerevisiae''6mA_T.thermophile''6mA_Tolypocladium''6mA_Xoc BLS256'

if __name__ == '__main__':
    TEST_PATH = "data/DNA_MS/tsv/5hmC/5hmC_M.musculus/test.tsv"
    MODEL_PATH = "result/5hmC_M.musculus/6mer/BERT, ACC[0.968].pt"
    OUTPUT_DIR = "CWGA/5hmC_M.musculus"
    config = config_init.get_config()
    FILM_Impact_Analysis(config, MODEL_PATH, TEST_PATH, OUTPUT_DIR)