# MEDNA-DFM
## MEDNA-DFM: A Dual-View FiLM-MoE Model for Explainable DNA Methylation Prediction

<div align="right">
  <a href="README_C.md">中文版本</a> | <a href="README.md"> English Version</a>
</div>

## 一、概述
本代码库为支持论文《MEDNA-DFM: A Dual-View FiLM-MoE Model for Explainable DNA Methylation Prediction》的开源实现，提供了 MEDNA-DFM 模型的核心架构及训练流程，旨在促进相关研究的复现、验证与进一步扩展。


## 二、结构详解
完整代码按模块化组件组织：

```
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

## 三、运行验证必要条件
### 1. 环境
- Python 3.9.18
- PyTorch 2.0.0（含CUDA 11.8）
- 依赖库：`transformers==4.18.0`

  安装方式：  
  ```bash
  conda env create -f MEDNA-DFM.yml
  ```

### 2. 预训练模型
MEDNA-DFM使用**DNABERT-6mer**和**DNABERT2**的微调版本作为基础架构。由于文件大小，这些权重并不包含在本仓库内。请从以下链接下载：
- [Download Fine-Tuned DNABERT Models](http://...)

### 3. 模型 Checkpoints (可选)
如果你希望跳过训练过程直接验证MEDNA-DFM，我们提供了最终的训练权重。你可以从以下链接下载：
- [Download MEDNA-DFM Checkpoints](http://...)


## 四、使用指南

**1. 模型准备:** 从上述链接下载微调股价模型（可选：训练模型权重）。 并将所下载文件放置如下位置：
- 放置DNABERT于 `MEDNA-DFM/fine_tuned_model`
- 放置最终训练权重于 `MEDNA-DFM/result`

**2. 配置参数:** 进入 `configuration/config_init.py` 设置所要训练的数据集, 超参数 (learning rate, batch size) 等.

**3. 训练模型:**
   ```bash
   python main/train.py
   ```
   
**验证论文结果、权重与代码**: 如果你想验证模型代码，模型最终的预测权重，跨物种预测结果以及外部数据集预测结果，请运行 `main/valid.py` (在 `valid.py` 中有更多信息)。



## 致谢

我们衷心感谢 iDNA-MS、iDNA-ABF、Methly-GP 等相关研究中开源模型的作者们，他们的工作为 DNA 甲基化预测领域的方法探索提供了宝贵的参考框架与开源资源，为本研究的模型设计与实验验证提供了重要借鉴。以及DNABERT & DNABERT2 作者，他们为 DNA 甲基化预测等下游任务提供了一个强大可靠的预训练模型。同时，感谢Hugging Face  `transformers` 的开发者，所有为表观基因组学计算方法开发提供开源工具和数据集的研究者，正是这些开放协作的学术实践推动了领域的快速发展。


## 联系方式
如有问题，请联系[tianchilu4-c@my.cityu.edu.hk],[heyi2023@lzu.edu.cn].