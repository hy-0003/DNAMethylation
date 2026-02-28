# Contrastive Weighted Gradient Attribution (CWGA)
Before running the CWGA analysis, please ensure that both your `fine_tuned_model` and `result` directories are properly configured and present in your workspace. Because the CWGA scripts utilize advanced analytical rendering, they require several additional libraries

> **Important Note Regarding Dependencies:** We are currently finalizing our comprehensive `environment.yml` file and will upload it to this repository shortly to facilitate one-click environment setup. 

> **In the meantime:** You can comfortably install any missing packages manually via `pip` or `conda` as you encounter `ModuleNotFoundError` prompts. Based on our extensive testing, the specific versions of these supplementary libraries do not significantly impact the final analytical results. Feel free to proceed with their latest stable versions!

## Directory Structure
```text
CWGA/
├── calculate_dim_grad_0mer.py
├── calculate_dim_grad_6mer.py
├── CWGA.py
├── Impact_Analysis.py
├── kmer_MotifLogo.py
└── README.md
```
## 1.Running the Impact_Analysis Analysis
`Impact_Analysis.py` is the pre analysise script. You can direct it to analyze specific datasets. Once the run is complete, the results are saved in the designated output directory (e.g., 6mA_D.melanogaster\Impact_Analysis.csv).

## 2.Running the CWGA Analysis
When we get the Impact_Analysis.csv, we can analyze dataset of Impact_Analysis.csv by running `CWGA.py`. Once the run is complete, the results are saved in the designated output directory (e.g., 6mA_D.melanogaster\CWGA_Top40_N100\0mer_contribution.txt and 6mer_contribution.txt). This directory will contain the sequence CWGA scores derived from grad's two path (DNABERT-6mer and DNABERT2).

## 3.Motif Discovery
Based on the CWGA results, we can make sequence with positive CWGA score as `positive.fasta` and sequence with negative CWGA score as `nagetive.fasta`. These fasta files are then submitted to STREME for downstream motif discovery, effectively isolating the true biological signals from the background noise.

## 4.Visualizing Motifs
To visualize the CWGA results, we provide the `kmer_MotifLogo.py` script. This script dynamically generates a sequence logo (often referred to as a KPLogo) highlighting the most informative base positions.
