# Reproduction of _Privacy-Preserving Action Recognition via Motion Difference Quantization_ 
This repository contains our reproduction and extension of the BDQ encoder from the paper _Privacy-Preserving Action Recognition via Motion Difference Quantization_. We replicate the results on the [**KTH**](https://www.csc.kth.se/cvap/actions/) dataset and additionally evaluate on [**IXMAS**](https://www.epfl.ch/labs/cvlab/data/data-ixmas10/). 

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
  - [Core Components](#core-components)
  - [Datasets \& Metadata](#datasets--metadata)
  - [Notebooks](#notebooks)
  - [Preprocessing Scripts](#preprocessing-scripts)
  - [Models \& Training](#models--training)
  - [Visualization \& Evaluation](#visualization--evaluation)
- [Run the Project](#run-the-project)
  - [Run On Kaggle](#run-on-kaggle)
  - [Run Locally](#run-locally)
- [Reproduce Figure 3](#reproduce-figure-3)

## Project Overview
We implemented the BDQ encoder consisting of three modules, Blur, Difference, and Quantization. The encoder is trained adversarially to retain action features while suppressing privacy-sensitive ones. 

## Directory Structure 
### Core Components
- `bdq_encoder/` 
  - `BDQ.py`: high-level wrapper for the BDQ encoder
  - `BDQ_disabled`: a passthrough version of the encoder that disables all BDQ transformations, used for ablation or baseline comparisons 
  - `blur.py`, `difference.py`, `quantization.py`: module-level definitions 

### Datasets & Metadata 
- Raw datasets available at: 
  - [KTH Dataset](https://www.csc.kth.se/cvap/actions/) 
  - [IXMAS Dataset](https://www.epfl.ch/labs/cvlab/data/data-ixmas10/) 
- `datasets/`
  - `datasets/KTH`: video files and `00sequences.txt` (metadata)
  - `datasets/IXAMS`: selected subset of IXMAS videos 
  - `datasets/util_kth.py`: action label map of the KTH dataset 
  - `datasets/ixmas_clips_6.json`, `datasets/kth_clips.json`: structured clip metadata generated from parsers 

### Notebooks 
- `notebooks/`
  - `notebooks/bdq-encoder.ipynb`: trains and evaluates the full BDQ encoder 
  -  `notebooks/bdq-no-encoder.ipynb`: baseline experiment without BDQ encoder 

### Preprocessing Scripts 
- `preprocess.py`: preprocessing pipeline 
- `raw_dataset_preprocess`: dataset-specific preprocessing tools 
  - `raw_dataset_preprocess/KTH_preprocess`
    - `.../KTH_preprocess/kth_parser.py`: parses `00sequences.txt` into structured KTH clip metadata 
  - `raw_dataset_preprocess/IXMAS_preprocess` 
    - `.../IXMAS_preprocess/IXMAS_720`: 720 manually selected representative frames (10 subject × 12 classes × 2 viewpoints × 3 takes) 
    - `.../IXMAS_preprocess/IXMAS_utils` 
      - `.../ixmas_extract_frame.py`: extracts and saves a representative frame from each IXMAS video 
      - `.../IXMAS_utils/ixmas_extract_vid.py`: locates and copies videos matching selected frame names (e.g., for `IXMAS_720/`) 
      - `.../IXMAS_utils/ixmas_6_vid.py`: filters and saves videos belonging to six selected actions 
      - `.../IXMAS_utils/ixmas_parser.py`: generates clip metadata from IXMAS video filenames 

### Models & Training 
- `loss.py`: loss functions for adversarial training 
- `training.py`: training and validation pipeline 
- `training_no_encoder.py`: an identical pipeline that bypasses BDQ transformations, used for ablation or baseline comparison 
- `action_recognition_model.py`: 3D ResNet-based action classifier 
- `privacy_attribute_prediction_model.py`: 2D ResNet-based identity classifier 

### Visualization & Evaluation 
- `visualization/`: logs and scripts for plotting results and evaluating BDQ performance 
  - `logs/`: TensorBoard-compatible log directories containing training and validation metrics 
  - `pics/`: generated plots 
  - `quantization_steps.py`: saves the initialized and learned quantization bin values during training 
  - `figure3_row1.py`: extracts final accuracy metrics from TensorBoard logs and plots the utility–privacy trade-off curve (Figure 3 row 1) 
  - `figure3_row2.py`: visualizes quantization steps using bin values (Figure 3 row 2) 
  - `train_val_acc.py`: plots training and validation accuracy curves over epochs for action and privacy prediction 
  - `val_acc_comp.py`: compares validation accuracy between different model variants (e.g., with and without BDQ) 

## Run the Project 
### Run On Kaggle 
1. Generate a GitHub Personal Access Token (PAT). 
2. Upload the notebook from [`/notebooks`](./notebooks) to Kaggle. 
3. Add the token as a Kaggle secret in your notebook: `Add-ons` -> `Secrets` -> `Add Secret`. 
4. Access the secret in the notebook by replacing the placeholder: 
  ```Python
  user_secrets = UserSecretsClient()
  token = user_secrets.get_secret("your_secrect_name") # replace with the actual name 
  ```
5. Specify the dataset to run: 
  ```Python
  %cd /kaggle/working/frmdl25-6/
  from training import main
  main('kth')
  ```
6. Enable GPU: `Settings` -> `Accelerator` -> select `GPU P100`. 

### Run Locally 

> **Note:** Training may be slow without a CUDA-enabled GPU. 

1. Install requirements: 
```bash
pip install -r requirements.txt
```
2. Run adversarial training: 
```bash
python training.py --dataset kth
# or
python training.py --dataset ixmas
```

## Reproduce Figure 3 

To replicate the two plots in Figure 3 from the original BDQ paper using saved logs, run the following commands: 
```bash
python visualization/figure3_row1.py
python visualization/figure3_row2.py
```
These scripts read from log files and generate the corresponding plots in [`visualization/pics/`](./visualization/pics/). 


## References 
[1] S. Kumawat and H. Nagahara, “Privacy-Preserving Action Recognition via Motion Difference Quantization,” Aug. 2022.

[2] C. Schuldt, I. Laptev, and B. Caputo, “Recognizing human actions: A local SVM approach,” in Proceedings of the 17th International Conference on Pattern Recognition, 2004. ICPR 2004., (Cambridge, UK), pp. 32–36 Vol.3, IEEE, 2004.

[3] D. Weinland, R. Ronfard, and E. Boyer, “Free viewpoint action recognition using motion history volumes,” Computer Vision and Image Understanding, vol. 104, pp. 249–257, Nov. 2006. 
