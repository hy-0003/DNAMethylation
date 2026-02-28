import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logomaker
import sys
import os

# 5hmC   '5hmC_H.sapiens','5hmC_M.musculus'
# 4mC    '4mC_C.equisetifolia','4mC_F.vesca','4mC_S.cerevisiae','4mC_Tolypocladium'
# 6mA    '6mA_A.thaliana''6mA_C.elegans''6mA_C.equisetifolia''6mA_D.melanogaster'
# 6mA    '6mA_F.vesca''6mA_H.sapiens''6mA_R.chinensis''6mA_S.cerevisiae''6mA_T.thermophile''6mA_Tolypocladium''6mA_Xoc BLS256'

# Please change your path
# K-mer score txt path
KMER_SCORES_FILE = r'CWGA\6mA_D.melanogaster\CWGA_Top40_N100\6mer_final_contribution_by_type.txt'
SEQUENCE_DATA_FILE = r'data\DNA_MS\tsv\6mA\6mA_D.melanogaster\test.tsv'
OUTPUT_LOGO_FILE = r'CWGA\Motif_Logo_6mer.svg'


KMER_SIZE = 6
SEQUENCE_LENGTH = 41


def load_kmer_scores(file_path):
    if not os.path.exists(file_path):
        print(f"Error: can not find files,please check path: {file_path}")
        sys.exit(1)
    try:
        kmer_scores_df = pd.read_csv(file_path, sep=r'\s+', engine='python')
        kmer_scores_df = kmer_scores_df[~kmer_scores_df['motif'].str.contains(r'\[|\]')]
        kmer_score_dict = pd.Series(
            kmer_scores_df.CAd.values,
            index=kmer_scores_df.motif
        ).to_dict()
        return kmer_score_dict
    except Exception as e:
        print(f"load k-mer score fail: {e}")
        sys.exit(1)

def load_positive_sequences(file_path):
    if not os.path.exists(file_path):
        print(f"Error: can not find file,please check path请: {file_path}")
        sys.exit(1)
    try:
        seq_df = pd.read_csv(file_path, sep='\t', header=0)
        seq_df['label'] = pd.to_numeric(seq_df['label'], errors='coerce')
        positive_sequences = seq_df[seq_df['label'] == 1]['text'].dropna().tolist()
        if not positive_sequences:
            print("Error: there is no positive sample (label == 1)")
            sys.exit(1)
        return positive_sequences
    except Exception as e:
        print(f"Load sequence fail: {e}")
        sys.exit(1)

def calculate_weighted_ppm(sequences, kmer_scores, seq_len, k):
    weighted_freq_matrix = np.zeros((4, seq_len))
    total_weights_per_position = np.zeros(seq_len)
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for seq in sequences:
        if isinstance(seq, str) and len(seq) == seq_len:
            for i in range(seq_len - k + 1):
                kmer = seq[i:i + k]
                score = max(0, kmer_scores.get(kmer, 0))
                if score > 0:
                    for j in range(k):
                        base = kmer[j]
                        pos = i + j
                        if base in base_map:
                            weighted_freq_matrix[base_map[base], pos] += score
                            total_weights_per_position[pos] += score
    ppm_matrix = np.zeros_like(weighted_freq_matrix)
    np.divide(weighted_freq_matrix, total_weights_per_position,
              out=ppm_matrix, where=total_weights_per_position != 0)
    ppm_df = pd.DataFrame(ppm_matrix, index=['A', 'C', 'G', 'T']).T
    ppm_df.columns.name = 'base'
    ppm_df.index.name = 'position'
    return ppm_df

def generate_motif_logo(ppm_df, output_file):
    if ppm_df is None or ppm_df.empty:
        print("Error: PPM is empty, can not plot Logo。")
        return
    try:
        info_df = logomaker.transform_matrix(
            ppm_df, from_type='probability', to_type='information'
        )
        seq_len = len(info_df)
        fig, ax = plt.subplots(figsize=(18, 3))
        logomaker.Logo(
            info_df,
            ax=ax,
            shade_below=.5,
            fade_below=.5,
            font_name='Arial',
            color_scheme={'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'},
            baseline_width=0
        )
        ax.set_ylabel('Information (bits)', fontsize=14, labelpad=15)
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(range(1, seq_len + 1), rotation=90, fontsize=14, fontweight='bold', color='red')
        ax.tick_params(axis='x', length=0)
        ax.set_ylim(-0.1, 2.1) 
        ax.set_xlim(-1, seq_len)
        for spine in ['top', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)
        plt.savefig(output_file, format='svg', bbox_inches='tight')
        print(f"SVG saved: {output_file}")
    except Exception as e:
        print(f"Ploting Logo fail: {e}")
        print(" please ensure pip 'logomaker': pip install logomaker")


def main():
    kmer_scores_dict = load_kmer_scores(KMER_SCORES_FILE)
    positive_sequences_list = load_positive_sequences(SEQUENCE_DATA_FILE)
    ppm_dataframe = calculate_weighted_ppm(
        sequences=positive_sequences_list,
        kmer_scores=kmer_scores_dict,
        seq_len=SEQUENCE_LENGTH,
        k=KMER_SIZE
    )
    generate_motif_logo(ppm_dataframe, OUTPUT_LOGO_FILE)


if __name__ == '__main__':
    main()