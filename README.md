# origin
This is a duplicate of the privat repository https://github.com/annekroon/publicsphere/ which is cleaned in terms of privacy sensitive data.

The work in writing the code for this repository was done by Wang Ngai (Justin) Yeung https://github.com/jyeungtin, Yufang (Arya) Liao  https://github.com/aryaliaoyf and Anne Kroon https://github.com/annekroon. 

Also see the code written by Sjoerd Stolwijk at https://github.com/cl-trier/TWON-Exploration/blob/master/27--Classify%20PublicSphere-paper_2024.ipynb which was used for the GLLM part of the paper accompanying this repo.

This repository provides the code used for the analysis for the following paper, please cite our paper if you use this repository:
Stolwijk, S. B., Boukes, M., Yeung, W. N., Liao, Y., Münker, S., Kroon, A. C., & Trilling, D. (2026). Can we use automated approaches to measure the quality of online political discussion? How to (not) measure interactivity, diversity, rationality, and incivility in online comments to the news. Communication Methods and Measures, 20(1), 1–25. https://doi.org/10.1080/19312458.2025.2553300

The citation is also available in [CITATION.cff](CITATION.cff); on GitHub, use the "Cite this repository" button in the sidebar to copy it in APA or BibTeX format.

The codebook used for the manual annotation baseline for the analysis in this repo can be found at: https://osf.io/cajqm/files/2g7ex.

UPDATE 2026: We have released newly trained versions of our transformer models on Hugging Face, collected in [CMM-2025-debate-quality](https://huggingface.co/collections/sjoerdstolwijk/cmm-2025-debate-quality-6ae1fd391952bf9b019e0af8); see below for more information.


# publicsphere
This project studies the deliberative quality of social media debates surrounding satire and traditional news.

You can replicate the result by running corresponding scripts if you have access to the original texts [not shared here out of privacy concerns]. 

## 0. Data & Preprocessing
### Raw data
The raw data includes:
- comments scraped from Twitter and Youtube
- manual coding results after fixing mistakes

### Final dataset 
Final datasets used for all measures and analyses should be placed in the *data* folder, including:
- **full_data.csv**: N = 3862
- **train.csv** and **test.csv**: split from full_data.csv with a ratio of 0.2

Each file includes: comments, meta information of comments, manual coding results, and aggregated manual coding. 

The three files could be reprocuded by running *0. Preprocessing*

Note that for GDPR and ethics reasons the full data is only available for research purposes on request.


## 1. Rule-based approaches
This section includes all scripts used for conducting rule-based measures.

**Data**: *full_data.csv*

**Outputs**: see *outputs/automated_results/*
  
### 1.1 Dictionaries
- Incivility: six dictioanries
- Diversity: MFD 1.0 and MFD 2.0

### 1.2 Rationality
- FK score
- Fake resource

### 1.3 Interactivity
- @ mention

### 1.4 eMFD
- Extended Moral Foundations Dictionary (eMFD, Hopp et al., 2020) 
  
### 1.5 Integrative complexity
- Integrative complexity score (Owens and Wedeking, 2011)
  
### 1.6 Formality (POS)
- Formality score (Heylighen and Dewaele, 2002)
  
## 2. Supervised machine learning
This section is the script used for conducting supervised machine learning.

**Data**: *train.csv* & *test.csv*

**Outputs**: see *outputs/automated_results/prediction_ML.csv* 

## 3. Combine all analyses
Here, the results of all analyses above are merged together. The new dataset will be used for comparison and evaluation.

**Data**: files in outputs/automated_results/

**Outputs**: data/Data_combination_test.csv

## 4. Evaluation
This section includes main evaluation and comparison presented in the paper, including correlation and precision, recall, and F1 score.

**Data**: Data_combination_test.csv 

**Outputs**: outputs/

### Precision, Recall, F1 score
e.g.,
<img width="462" alt="截屏2023-08-08 20 36 11" src="https://github.com/annekroon/publicsphere/assets/105493457/a9ac893a-4f5a-4de0-9bba-1f079ecbbb7c">

### Correlation
e.g., 
![Correlations: all measures in Incivility](https://github.com/annekroon/publicsphere/assets/105493457/ee078555-4d4a-481c-879e-2c27db7f0a7a)

## 5. Improvements
This section includes all exploratory analyses and improvements for automated measures. 

### 5.1 Dictionaries as input for machine learning
**Data**: *train.csv* & *test.csv*

**Outputs**: 

### 5.2 Merging dictionaries (at_least_one & majority_vote)
**Data**: Data_combination_test.csv

**Outputs**: 

visualization bar tables: 
    ![Merging dictionaries   class 1 (all)](https://github.com/annekroon/publicsphere/assets/105493457/e3de936d-5230-4301-af98-5064e37f9a3f)
    
### 5.3 Distributed Dictionaries Representation (DDR) method
**Data**: *test.csv*

**Outputs**: 
<img width="670" alt="截屏2023-08-08 20 38 34" src="https://github.com/annekroon/publicsphere/assets/105493457/5da1f541-d595-4fb2-90b7-11da2b5e1b10">


### Sampling test
Please find all data and plots in the Monte Carlo folder. Script is in the root. 

### Threshold test
**Data**: *Data_combination_test.csv*

**Output**: Find results in the notebook
  
### Additional dataset
**Output**: Find results in the notebook

### UPDATE 2026 - BERT model release:
We have used our manual coded data to retrain a set of BERT_base_uncased transformer models for seven concepts included in our manual coded baseline: Conservative, Liberal, Has Opinion, Hatelist-focussed, Incivility, Interactivity and Rationality. The "BERT_model_release_2026" folder contains the code used for training and evaluating these models as well as a report of their performance against the test set. For more information on the manual annotations underlying these models see: https://osf.io/cajqm/files/2g7ex

The models are available on Hugging Face in the [CMM-2025-debate-quality](https://huggingface.co/collections/sjoerdstolwijk/cmm-2025-debate-quality-6ae1fd391952bf9b019e0af8) collection:

| Concept | Model |
|---|---|
| Conservative | [bert-base-uncased-CMM-2025-debate-quality-Conservative](https://huggingface.co/sjoerdstolwijk/bert-base-uncased-CMM-2025-debate-quality-Conservative) |
| Liberal | [bert-base-uncased-CMM-2025-debate-quality-Liberal](https://huggingface.co/sjoerdstolwijk/bert-base-uncased-CMM-2025-debate-quality-Liberal) |
| Incivility | [bert-base-uncased-CMM-2025-debate-quality-Incivility](https://huggingface.co/sjoerdstolwijk/bert-base-uncased-CMM-2025-debate-quality-Incivility) |
| Interactivity | [bert-base-uncased-CMM-2025-debate-quality-Interactivity](https://huggingface.co/sjoerdstolwijk/bert-base-uncased-CMM-2025-debate-quality-Interactivity) |
| Rationality | [bert-base-uncased-CMM-2025-debate-quality-Rationality](https://huggingface.co/sjoerdstolwijk/bert-base-uncased-CMM-2025-debate-quality-Rationality) |
| Has-Opinion | [bert-base-uncased-CMM-2025-debate-quality-Has-Opinion](https://huggingface.co/sjoerdstolwijk/bert-base-uncased-CMM-2025-debate-quality-Has-Opinion) |
| Hatelist-Focused | [bert-base-uncased-CMM-2025-debate-quality-Hatelist-Focused](https://huggingface.co/sjoerdstolwijk/bert-base-uncased-CMM-2025-debate-quality-Hatelist-Focused) |

## AI notice
Claude Opus 5.5 (`claude-opus-5-5`) was used to help with the 2026 BERT model release:
- the Hugging Face upload notebook (`BERT_model_release_2026/models/huggingface.ipynb`), the model cards and the readable labels in each model's config;
- two fixes in the training and evaluation scripts: the argument order of `classification_report` in `scripts.train.py`, and the label lookup in `scripts.evaluate.py`.

The models were trained, and the original analyses written, without AI assistance.
