import os, sys, torch, gc, transformers
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from types import SimpleNamespace
from collections import defaultdict
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"The project path has been add to sys.path: {project_root}")
try:
    # try import 
    from configuration import config_init
    from module.Fusion_module import Fusion
    from util import util_file
    # DNABERT for CAD
    import DNAbert_model2
except ImportError as e:
    print(f"Error：can not import necessary module. Please ensure DNAbert_model2 and configuration exist under {project_root}")
    print(f"Error Information: {e}")
    sys.exit(1)
transformers.logging.set_verbosity(transformers.logging.ERROR)


def load_and_filter_samples(
    TEST_PATH: str, 
    MODEL_PATH: str, 
    config, 
    batch_size: int = 128, 
    confidence_threshold: float = 0.9
):
    # --- 1. Load model and Data (for slecetion of sample) ---
    print("Step1: Load model and Data (Foselcetion of sample)...")
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    model_for_filtering = Fusion(config) 
    model_for_filtering.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model_for_filtering.to(device)
    model_for_filtering.eval()
    sequences, labels = util_file.load_tsv_format_data(TEST_PATH)
    print(f"Load sucess. Total {len(sequences)} Sequence.")
    
    # --- 2. Slect high confidence sample ---
    print(f"Step2: Slect high confidence sample (Confidence > {confidence_threshold}) ...")
    correct_positive_samples = []
    correct_negative_samples = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            seq_batch = sequences[i:i + batch_size]
            label_batch = labels[i:i+batch_size]
            outputs, _ = model_for_filtering(seq_batch)
            probs = torch.softmax(outputs, dim=1)
            for j in range(len(seq_batch)):
                confidence_pos = probs[j, 1].item()
                if label_batch[j] == 1 and confidence_pos > confidence_threshold:
                    correct_positive_samples.append({
                        "sequence": seq_batch[j], 
                        "confidence": confidence_pos
                    })
                confidence_neg = probs[j, 0].item()
                if label_batch[j] == 0 and confidence_neg > confidence_threshold:
                    correct_negative_samples.append({
                        "sequence": seq_batch[j], 
                        "confidence": confidence_neg
                    })
    correct_positive_samples.sort(key=lambda x: x['confidence'], reverse=True)
    correct_negative_samples.sort(key=lambda x: x['confidence'], reverse=True)
    target_sequences_pos = [s['sequence'] for s in correct_positive_samples]
    target_sequences_neg = [s['sequence'] for s in correct_negative_samples]
    print(f"Had slected {len(target_sequences_pos)} positive sample (Conf > {confidence_threshold}) and {len(target_sequences_neg)} negative sample (Conf > {confidence_threshold}).")
    
    # --- 3. Kill model ---
    print("Step3: Kill slection model, return sample...")
    del model_for_filtering, sequences, labels, correct_positive_samples, correct_negative_samples, probs, outputs
    gc.collect()
    torch.cuda.empty_cache()
    print("sucess")
    return target_sequences_pos, target_sequences_neg


def get_cls_attribution_scores(model, dna_sequence: str):
    try:
        attentions, tokens = model.get_attention_maps(dna_sequence)
        if attentions is None or tokens is None:
            print(f"Pay Attention: Sequence {dna_sequence[:15]}... can not get attention, pass")
            return {}
        last_layer_attentions = attentions[-1]
        avg_head_attentions = last_layer_attentions.mean(dim=1) 
        avg_head_attentions = avg_head_attentions.squeeze(0) 
        cls_attention_row = avg_head_attentions[0, :] 
        cls_scores_np = cls_attention_row.detach().cpu().numpy()
        try:
            pad_index = tokens.index('[PAD]')
            tokens = tokens[:pad_index]
            cls_scores_np = cls_scores_np[:pad_index]
        except ValueError:
            pass 
        tokens_cleaned = []
        scores_cleaned = []
        try:
            sep_index = tokens.index('[SEP]')
            tokens_cleaned = tokens[1:sep_index]
            scores_cleaned = cls_scores_np[1:sep_index]
        except ValueError:
            if tokens[0] == '[CLS]' and tokens[-1] == '[SEP]':
                tokens_cleaned = tokens[1:-1]
                scores_cleaned = cls_scores_np[1:-1]
            else:
                tokens_cleaned = tokens
                scores_cleaned = cls_scores_np
        if len(tokens_cleaned) == 0:
            return {}
        return dict(zip(tokens_cleaned, scores_cleaned))
    except Exception as e:
        print(f"Error: While Processing {dna_sequence[:15]} fail: {e}, Pass")
        return {}


def save_aggregated_scores_to_txt(scores_dict, filepath: str):
    if not scores_dict:
        print(f"Error: dict is empty, can not save to {filepath}")
        return
    df = pd.DataFrame(scores_dict.items(), columns=['token', 'aggregated_attention'])
    df.sort_values(by='aggregated_attention', ascending=False, inplace=True)
    df.to_csv(filepath, sep='\t', index=False, float_format='%.8f')
    print(f"Saved attentin score in: {filepath}")


def calculate_cohens_d(x, y):
    # Compute Cohen's d
    # Check at least 2 sample
    if len(x) < 2 or len(y) < 2:
        return 0.0, np.nan
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    if std_x == 0 and std_y == 0:
        return 0.0, 1.0
    pooled_std = np.sqrt(((nx - 1) * std_x**2 + (ny - 1) * std_y**2) / dof)
    if pooled_std == 0:
        d = 0.0
    else:
        d = (mean_x - mean_y) / pooled_std
    ttest_result = ttest_ind(x, y, equal_var=False)
    p_value = ttest_result.pvalue
    return d, p_value


def run_attention_analysis_for_kmer(
    kmer_setting: int, 
    sequences_pos: list, 
    sequences_neg: list, 
    output_dir: str
):
    # Run CAd (Contrastive Attention d-score)
    print(f"=== Step B: Start KMER={kmer_setting}'s CAD Analysis ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- 1. Load DNABERT model ---
    print(f"Step B.1: Load KMER={kmer_setting}'s DNABERT model...")
    try:
        config_attn = SimpleNamespace()
        config_attn.kmer = kmer_setting
        config_attn.cuda = torch.cuda.is_available()
        model_attn = DNAbert_model2.BERT(config_attn)
        model_attn.to(device)
        model_attn.eval()
        print(f"Load sucess. Path: {model_attn.pretrainpath}")
    except Exception as e:
        print(f"Error: Load KMER={kmer_setting}'s DNABERT fail: {e}")
        return
    
    # --- 2. Analysis positive sample ---
    print(f"\nStep B.2: Analysis {len(sequences_pos)} high confidence positive sample (KMER={kmer_setting})...")
    motif_scores_pos = defaultdict(list)
    for i, seq in enumerate(sequences_pos):
        if (i+1) % 50 == 0: 
            print(f"     ... processing {i+1}/{len(sequences_pos)}")
        scores_dict = get_cls_attribution_scores(model_attn, seq)
        for token, score in scores_dict.items():
            motif_scores_pos[token].append(score) 
    print("Finished")

    # --- 3. Analysis negative sample ---
    print(f"\nStep B.3: Analysis {len(sequences_neg)} high confidence negative sample (KMER={kmer_setting})...")
    motif_scores_neg = defaultdict(list)
    for i, seq in enumerate(sequences_neg):
        if (i+1) % 50 == 0:
            print(f"     ... processing {i+1}/{len(sequences_neg)}")
        scores_dict = get_cls_attribution_scores(model_attn, seq)
        for token, score in scores_dict.items():
            motif_scores_neg[token].append(score) 
    print("Finished")

    # --- 4. compute CAd and p-value ---
    print(f"\nStep B.4: compute CAD score (KMER={kmer_setting})...")
    all_motifs = set(motif_scores_pos.keys()) | set(motif_scores_neg.keys())
    statistical_data = []
    MIN_SAMPLES = 3
    for motif in all_motifs:
        pos_list = motif_scores_pos.get(motif, [])
        neg_list = motif_scores_neg.get(motif, [])
        n_pos = len(pos_list)
        n_neg = len(neg_list)
        if n_pos < MIN_SAMPLES or n_neg < MIN_SAMPLES:
            continue
        mean_pos = np.mean(pos_list)
        mean_neg = np.mean(neg_list)
        d_value, p_value = calculate_cohens_d(pos_list, neg_list)
        statistical_data.append({
            'motif': motif,
            'CAd': d_value,
            'p_value': p_value,
            'mean_pos_attention': mean_pos,
            'mean_neg_attention': mean_neg,
            'n_pos_samples': n_pos,
            'n_neg_samples': n_neg
        })
    if not statistical_data:
        print("Error: fail, pass save.")
        del model_attn, motif_scores_pos, motif_scores_neg
        gc.collect()
        torch.cuda.empty_cache()
        return

    df_stats = pd.DataFrame(statistical_data)
    df_stats.sort_values(by='CAd', ascending=False, inplace=True)
    kmer_str = f"{kmer_setting}mer"
    if kmer_setting == 0:
        kmer_str = "0mer"
    stats_filename = os.path.join(output_dir, f"{kmer_str}_CAD.txt")
    df_stats.to_csv(stats_filename, sep='\t', index=False, float_format='%.8f')
    print(f"Saved CAD score to: {stats_filename}")

    # --- 5. clean model ---
    del model_attn, motif_scores_pos, motif_scores_neg, df_stats, statistical_data
    gc.collect()
    torch.cuda.empty_cache()
    print(f"KMER={kmer_setting} model has been killed")


if __name__ == "__main__":
    TEST_PATH = "data/DNA_MS/tsv/6mA/6mA_D.melanogaster/test.tsv"
    MODEL_PATH = "result/6mA_D.melanogaster/6mer/BERT, ACC[0.930].pt"
    ATTENTION_OUTPUT_DIR = os.path.join(current_dir, "6mA_D.melanogaster/CAD_Result")

    os.makedirs(ATTENTION_OUTPUT_DIR, exist_ok=True)
    try:
        config = config_init.get_config()
    except Exception as e:
        print(f"Error: load config_init.get_config() fail: {e}")
        sys.exit(1)
    print("=== Step A: load and slect high confidence sample ===")
    target_sequences_pos, target_sequences_neg = load_and_filter_samples(
        TEST_PATH=TEST_PATH,
        MODEL_PATH=MODEL_PATH,
        config=config,
        batch_size=128,
        confidence_threshold=0.9
    )
    if not target_sequences_pos or not target_sequences_neg:
        print("Error：can not slect enough sample.")
        sys.exit(1)

    
    run_attention_analysis_for_kmer(
        kmer_setting=6, 
        sequences_pos=target_sequences_pos,
        sequences_neg=target_sequences_neg,
        output_dir=ATTENTION_OUTPUT_DIR
    )
    run_attention_analysis_for_kmer(
        kmer_setting=0, 
        sequences_pos=target_sequences_pos,
        sequences_neg=target_sequences_neg,
        output_dir=ATTENTION_OUTPUT_DIR
    )
    print("\n" + "="*80)
    print("=== All the result Saved sucessfully！ ===")
    print(f"Path: {ATTENTION_OUTPUT_DIR}")
    print("="*80)