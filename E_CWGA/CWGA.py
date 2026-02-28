import os, sys, torch, gc, queue
import pandas as pd
from collections import defaultdict
from multiprocessing import Process, Manager

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from configuration import config_init
from module.Fusion_module import Fusion
from util import util_file
from calculate_dim_grad_0mer import calculate_token_per_dimension_contribution_0mer
from calculate_dim_grad_6mer import calculate_token_per_dimension_contribution_6mer
import transformers
transformers.logging.set_verbosity(transformers.logging.ERROR)


def process_dimension_worker(
    target_dim, 
    weight_w_i, 
    sequences_pos, 
    sequences_neg,
    model_path, 
    config, 
    grad_batch_size,
    result_queue
):
    try:
        pid = os.getpid()
        print(f"\n[Process {pid}] start processing {target_dim} (weight: {weight_w_i:.4f})")
        device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
        model = Fusion(config)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.to(device)
        model.eval()
        print(f"    [Process {pid} | 0-mer] precessing {target_dim}...")
        delta_0mer = calculate_token_per_dimension_contribution_0mer(
            model, target_dim, sequences_pos, sequences_neg, grad_batch_size
        )
        print(f"    [Process {pid} | 6-mer] processing {target_dim}...")
        delta_6mer = calculate_token_per_dimension_contribution_6mer(
            model, target_dim, sequences_pos, sequences_neg, grad_batch_size
        )
        result_queue.put((target_dim, weight_w_i, delta_0mer, delta_6mer))
        print(f"[Process {pid}] dim {target_dim} finish.")

    except Exception as e:
        print(f"[Process {pid}] processing {target_dim} wrong: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put((target_dim, weight_w_i, None, None))
    finally:
        del model
        torch.cuda.empty_cache()


def run_full_analysis():
    # 5hmC   '5hmC_H.sapiens','5hmC_M.musculus'
    # 4mC    '4mC_C.equisetifolia','4mC_F.vesca','4mC_S.cerevisiae','4mC_Tolypocladium'
    # 6mA    '6mA_A.thaliana''6mA_C.elegans''6mA_C.equisetifolia''6mA_D.melanogaster'
    # 6mA    '6mA_F.vesca''6mA_H.sapiens''6mA_R.chinensis''6mA_S.cerevisiae''6mA_T.thermophile''6mA_Tolypocladium''6mA_Xoc BLS256'

    TEST_PATH = "data/DNA_MS/tsv/6mA/6mA_D.melanogaster/test.tsv"
    MODEL_PATH = "result/6mA_D.melanogaster/6mer/BERT, ACC[0.930].pt"
    BASE_OUTPUT_DIR = "CWGA/6mA_D.melanogaster"
    TOP_N_DIMS_FOR_AGGREGATION = 40
    NUM_SAMPLES_TO_ANALYZE = 100
    GRAD_CALCULATION_BATCH_SIZE = 5
    AGGREGATED_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"CWGA_Top{TOP_N_DIMS_FOR_AGGREGATION}_N{NUM_SAMPLES_TO_ANALYZE}")
    os.makedirs(AGGREGATED_OUTPUT_DIR, exist_ok=True)
    config = config_init.get_config()

    
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    model_for_filtering = Fusion(config) 
    model_for_filtering.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model_for_filtering.to(device)
    model_for_filtering.eval()
    sequences, labels = util_file.load_tsv_format_data(TEST_PATH)
    correct_positive_samples, correct_negative_samples = [], []
    with torch.no_grad():
        batch_size = 128
        for i in range(0, len(sequences), batch_size):
            seq_batch = sequences[i:i + batch_size]
            label_batch = labels[i:i+batch_size]
            outputs, _ = model_for_filtering(seq_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            for j in range(len(seq_batch)):
                if preds[j] == 1 and label_batch[j] == 1:
                    correct_positive_samples.append({"sequence": seq_batch[j], "confidence": probs[j, 1].item()})
                elif preds[j] == 0 and label_batch[j] == 0:
                    correct_negative_samples.append({"sequence": seq_batch[j], "confidence": probs[j, 0].item()})
    correct_positive_samples.sort(key=lambda x: x['confidence'], reverse=True)
    correct_negative_samples.sort(key=lambda x: x['confidence'], reverse=True)
    if NUM_SAMPLES_TO_ANALYZE is not None:
        target_sequences_pos = [s['sequence'] for s in correct_positive_samples[:NUM_SAMPLES_TO_ANALYZE]]
        target_sequences_neg = [s['sequence'] for s in correct_negative_samples[:NUM_SAMPLES_TO_ANALYZE]]
    else:
        target_sequences_pos = [s['sequence'] for s in correct_positive_samples]
        target_sequences_neg = [s['sequence'] for s in correct_negative_samples]


    del model_for_filtering, sequences, labels 
    gc.collect()
    torch.cuda.empty_cache()


    vector_analysis_path = os.path.join(BASE_OUTPUT_DIR, "Impact_Analysis.csv")
    if not os.path.exists(vector_analysis_path):
        print(f"Error: can not find file {vector_analysis_path}")
        return
    df_va = pd.read_csv(vector_analysis_path)
    top_dims_df = df_va.head(TOP_N_DIMS_FOR_AGGREGATION)
    summed_scores_0mer_by_type = defaultdict(float)
    summed_scores_6mer_by_type = defaultdict(float)
    manager = Manager()
    result_queue = manager.Queue()
    for i, (_, row) in enumerate(top_dims_df.iterrows(), start=1):
        target_dim = int(row['维度'])
        weight_w_i = row['科恩d值']
        p = Process(
            target=process_dimension_worker,
            args=(
                target_dim, 
                weight_w_i, 
                target_sequences_pos,
                target_sequences_neg,
                MODEL_PATH,          
                config,            
                GRAD_CALCULATION_BATCH_SIZE,
                result_queue 
            )
        )
        p.start()
        try:
            timeout_seconds = 1800 
            dim_res, w_i, delta_0mer, delta_6mer = result_queue.get(timeout=timeout_seconds)
            p.join()
            p.close()
            if delta_0mer is None or delta_6mer is None:
                print(f"  - Error: dim {dim_res} compute fail,pass")
                continue
            for token, grad_diff in delta_0mer.items():
                summed_scores_0mer_by_type[token] += w_i * grad_diff
            for kmer, grad_diff in delta_6mer.items():
                summed_scores_6mer_by_type[kmer] += w_i * grad_diff
            del delta_0mer, delta_6mer
            gc.collect()

        except queue.Empty:
            print(f"  - Error: dim {target_dim} don not in {timeout_seconds}s return the result！")
            print("  - main process is killing the subprocess...")
            p.terminate()
            p.join()
            p.close()

    
    manager.shutdown()

    # --- 0-mer ---
    df_0mer = pd.DataFrame(summed_scores_0mer_by_type.items(), columns=['tokens', 'summed_contribution'])
    max_abs_val = 0.0
    if not df_0mer.empty:
        max_abs_val = df_0mer['summed_contribution'].abs().max()
    df_0mer['scaled_contribution'] = df_0mer['summed_contribution'] / max_abs_val if max_abs_val > 1e-9 else 0.0
    txt_path_0mer = os.path.join(AGGREGATED_OUTPUT_DIR, "0mer_contribution.txt")
    df_0mer.sort_values(by='scaled_contribution', ascending=False, inplace=True)
    df_0mer.to_csv(txt_path_0mer, sep='\t', index=False, float_format='%.8f')

    # --- 6-mer ---
    df_6mer = pd.DataFrame(summed_scores_6mer_by_type.items(), columns=['tokens', 'summed_contribution'])
    max_abs_val = 0.0
    if not df_6mer.empty:
        max_abs_val = df_6mer['summed_contribution'].abs().max()
    df_6mer['scaled_contribution'] = df_6mer['summed_contribution'] / max_abs_val if max_abs_val > 1e-9 else 0.0
    txt_path_6mer = os.path.join(AGGREGATED_OUTPUT_DIR, "6mer_contribution.txt")
    df_6mer.sort_values(by='scaled_contribution', ascending=False, inplace=True)
    df_6mer.to_csv(txt_path_6mer, sep='\t', index=False, float_format='%.8f')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    run_full_analysis()