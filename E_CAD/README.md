# Contrastive Attention Cohen’s d (CAD)
Before running the CAD analysis, please ensure that both your `fine_tuned_model` and `result` directories are properly configured and present in your workspace. Because the CAD and Motif visualization scripts utilize advanced analytical rendering, they require several additional libraries

> **Important Note Regarding Dependencies:** We are currently finalizing our comprehensive `environment.yml` file and will upload it to this repository shortly to facilitate one-click environment setup. 

> **In the meantime:** You can comfortably install any missing packages manually via `pip` or `conda` as you encounter `ModuleNotFoundError` prompts. Based on our extensive testing, the specific versions of these supplementary libraries do not significantly impact the final analytical results. Feel free to proceed with their latest stable versions!

## Directory Structure
```text
CAD/
├── CAD.py
├── DNAbert_model2.py
├── kmer_MotifLogo.py
└── README.md
```
## 1.Running the CAD Analysis
`CAD.py` is the primary executable script. You can direct it to analyze specific datasets. Once the run is complete, the results are saved in the designated output directory (e.g., 6mA_D.melanogaster\CAD_Result). This directory will contain the sequence CAD scores derived from DNABERT-6mer and DNABERT2.

## 2.Motif Discovery
Based on the CAD results, we can make sequence with positive CAD score as `positive.fasta` and sequence with negative CAD score as `nagetive.fasta`. These fasta files are then submitted to STREME for downstream motif discovery, effectively isolating the true biological signals from the background noise.

## 3.Visualizing Motifs
To visualize the 6-mer CAD results, we provide the `kmer_MotifLogo.py` script. This script dynamically generates a sequence logo (often referred to as a KPLogo) highlighting the most informative base positions.
