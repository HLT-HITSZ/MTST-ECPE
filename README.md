# Introduction
This repository was used in our paper:  
  
**“Multi-task Sequence Tagging for Emotion-Cause Pair Extraction via Tag Distribution Refinement”**  
Chuang Fan, Chaofa Yuan, Lin Gui, Yue Zhang, Ruifeng Xu. TASLP 2021
  
Please cite our paper if you use this code.  
# Prerequisites
Python 3.7  
[Pytorch](https://pytorch.org/) 1.1.0  
[CUDA](https://developer.nvidia.com/cuda-10.0-download-archive) 10.0  
BERT - Our bert model is adapted from this implementation: https://github.com/huggingface/pytorch-pretrained-BERT  
# Descriptions
**20-fold-data-splits** - A dir where contains data splits following [Fan et al.](https://www.aclweb.org/anthology/2020.acl-main.342.pdf)
  * ```train.pkl```: A list where contains two items. train\[0\] is a list of document and train\[1\] is a list of the correspondding emotion-cause pairs. For example, train\[0\]\[0\]="*Last week, I lost my phone where shopping, I feel sad now*", then train\[1\]\[0\]=\[(2, 1)\].  
  * ```valid.pkl```: Similar to train.pkl.  
  * ```test.pkl```: Similar to train.pkl.
  
**10-fold-data-splits** - A dir where contains data splits following [Xia and Ding.](https://www.aclweb.org/anthology/2020.acl-main.342.pdf) The data format is the same as 20-fold-data-splits.
    
**bert-base-chinese** - Put the download Pytorch bert model here. 

**Utils** - A dir where contains several python scripts used in this code.  
* ```Evaluation.py```: Used to evaluate the performance of the proposed model.  
* ```Metrics.py```: Metrics for emotion extraction, cause extraction and emotion-cause pair extractions.  
* ```PrepareData.py```: The scipt for preparing data.  

```Config.py``` - The script holds all the model configuration.  
```Modules.py``` - The script where contains the proposed multi-task sequence tagging model.  
```Run.py``` - The main script to train and evaluate the proposed model on different splits.  
# Usage
python3 Run.py
