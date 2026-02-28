# MEDNA-DFM
## MEDNA-DFM: A Dual-View FiLM-MoE Model for Explainable DNA Methylation Prediction

<div align="right">
  <a href="README_C.md">中文版本</a> | <a href="README.md"> English Version</a>
</div>

## I. Overview
This repository provides the open-source implementation for the paper "MEDNA-DFM: A Dual-View FiLM-MoE Model for Explainable DNA Methylation Prediction". It includes the core architecture and training pipeline of the MEDNA-DFM model, aiming to facilitate the reproduction, validation, and further extension of related research.

## II. Directory Structure
The complete code is organized into modular components:

```text
MEDNA-DFM
├── configuration
│   └── config_init.py
├── data
|   ├── DNA_MS/tsv
|   |          ├── 4mC/...
|   |          ├── 5hmC/...
|   |          └── 6mA/...
|   └── external/test.tsv
├── E_CAD          
├── E_CWGA
├── fine_tuned_model
|   ├── 4mC/...
|   ├── 5hmC/...
|   └── 6mA/...
├── frame          
|   ├── IOManager.py        
|   ├── Learner.py          
│   ├── ModelManager.py  
│   └── Visualizer.py       
├── main              
│   ├── train.py 
│   └── valid.py 
├── module                
|   ├── Adversarial_module.py
|   ├── Dataprocess_model.py
│   ├── DNABERT_module.py      
│   └── Fusion_module.py 
├── result
├── util                    
│   └── util_file.py
├── MEDNA-DFM.yml
├── README_C.md 
└── README.md
```

## III. Prerequisites for Execution and Validation
### 1. Environment
- Python 3.9.18
- PyTorch 2.0.0 (including CUDA 11.8)
- Dependencies: `transformers==4.18.0`

  Installation:  
  ```bash
  conda env create -f MEDNA-DFM.yml
  ```

### 2. Pre-trained Models
MEDNA-DFM utilizes fine-tuned versions of **DNABERT-6mer** and **DNABERT2** as its foundational backbone. Due to file size constraints, these models are not included directly in this repository. Please download them from the link below:
- [Download Fine-Tuned DNABERT Models](https://huggingface.co/hy-0003/weight1)

### 3. Trained Model Checkpoints (Optional)
If you wish to skip the training process and directly evaluate MEDNA-DFM, we provide the final trained weights of our model across different datasets. You can download these prediction weights from the following link:
- [Download MEDNA-DFM Checkpoints](https://huggingface.co/hy-0003/weight2)

## IV. Usage Guide

**1. Prepare the Models:** Download the fine-tuned backbone models (and optionally the trained model checkpoints) from the links above. Extract and place them into their respective directories:
- Put the DNABERT models in `MEDNA-DFM/fine_tuned_model`.
- Put the final prediction weights in `MEDNA-DFM/result`.

**2. Configure parameters:** Modify `configuration/config_init.py` to set up the datasets, hyperparameters (learning rate, batch size), and paths.

**3. Train the Model:**
   ```bash
   python main/train.py
   ```
   
**Validation of results, weights, and code**: If you want to verify the feasibility of the model's code, the correctness of the model's prediction weights, the cross-species results from the paper, and the validation results on external datasets, please run `main/valid.py` (see `valid.py` for more information).

**CAD and CWGA Analysis**: For advanced analyses using CAD and CWGA, please navigate to their specific folders. Because these modules require additional dependencies, we have provided dedicated README.md files within each directory. Ensure you read through these localized instructions to properly configure your environment before running the scripts.

## Demo and Website
For the easiest experience, we will host our models on Hugging Face latter. You can integrate them into your code with just a few lines using the `transformers` library, or try our interactive web demo directly in your browser without any installation:

**[Try the Interactive Web Demo (Hugging Face Space)](https://huggingface.co/spaces/hy-0003/MEDNA-DFM-Web)**


## Acknowledgements

We sincerely thank the authors of open-source models in related studies such as iDNA-MS, iDNA-ABF, and Methly-GP. Their work has provided valuable reference frameworks and open-source resources for methodological exploration in the field of DNA methylation prediction, offering important insights for the model design and experimental validation of this study. We also thank the authors of DNABERT & DNABERT2, who provided a powerful and reliable pre-trained model for downstream tasks such as DNA methylation prediction. At the same time, we thank the developers of Hugging Face `transformers`, and all researchers who have provided open-source tools and datasets for the development of computational methods in epigenomics. It is these open and collaborative academic practices that drive the rapid development of the field.

## Contact
If you have any questions, please contact [tianchilu4-c@my.cityu.edu.hk],[heyi2023@lzu.edu.cn].
