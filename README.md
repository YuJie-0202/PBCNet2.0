# Advancing Ligand Binding Affinity Prediction with Cartesian Tensor-Based Deep Learning (Under Review)
Implementation of PBCNet2.0, by Jie Yu and Xia Shen.

This repository contains all code, instructions and model weights necessary to make predictions of relative binding affinity by PBCNet2.0, eval PBCNet2.0 or to retrain a new model.
![image](https://github.com/user-attachments/assets/270d9a49-93ff-47e0-b027-c09ff17943c2)




### 0. Environment
Our test environment information:
```
System Info
---------------------------------------     
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 18.04.6 LTS
Release:        18.04
Codename:       bionic
ldd --version
ldd (Ubuntu GLIBC 2.27-3ubuntu1.6) 2.27
---------------------------------------
GPU Info
NVIDIA-SMI 470.57.02
Driver Version: 470.57.02
CUDA Version: 11.4
```

You can follow the instructions to setup the environment (Please select the appropriate version of the software based on your hardware during installation):
```
conda create --name pbcnet python=3.8
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
```
pip install dgl==1.0.2 -f https://data.dgl.ai/wheels/cu113/repo.html --no-deps
pip install numpy
pip install pandas
pip install packaging
pip install PyYAML
pip install pydantic
pip install scipy
pip install networkx psutil tqdm
pip install scikit-learn
```


### 1. Model weights and code

1. The file ``./PBCNet2.0.pth`` contains the model weights. Load it using:

```
    # python
    model = torch.load(f"{code_path}/PBCNet2.0.pth", map_location=torch.device('cuda:1'), weights_only=False)
```

    Note: Adjust code_path and GPU device settings as needed.

2. The ``./model_code`` directory provides the implementation of PBCNet2.0.



### 2. Reproducing Paper Results
The ``./Results_in_paper`` directory contains code to reproduce all results reported in our paper.
Note: Update relevant paths during execution.

### 3. Data set
1. Test Data: Included in the ``./data`` directory.
2. Training Data: Available at Zendo(https://zenodo.org/records/15656365). To retrain PBCNet2.0:

   2.1 Download the data.
   2.2 Preprocess SDF/PDB files into PKL format using ``./Graph2pickle.py`` for model compatibility.

### 4. Making prediction with PBCNet2.0
See the ``./case/try.ipynb`` notebook for a step-by-step prediction example. Provide ligand (SDF) and protein (PDB) files to run predictions.


### 5. Fintuning the PBNCet2.0
Execute ``./model_code/run_finetune.sh`` to fine-tune PBCNet2.0 using the FEP dataset.
Custom Data: Replace paths in relevant files to use your own data.

### 6. Retraining the PBNCet2.0
1. Download training data and input file ``training_clip_862W.zip`` from Zendo(https://zenodo.org/records/15656365) and convert SDF/PDB files to PKL format using ``./Graph2pickle.py``.

2. Run ``./model_code/run_train.sh``.。

### 7. License
MIT
