# Reproduction of _Privacy-Preserving Action Recognition via Motion Difference Quantization_ 
This repository contains our reproduction and extension of the BDQ encoder from the paper _Privacy-Preserving Action Recognition via Motion Difference Quantization_
We replicate the results on the [**KTH**](https://www.csc.kth.se/cvap/actions/) dataset and additionally evaluate on [**IXMAS**](https://www.epfl.ch/labs/cvlab/data/data-ixmas10/). 

## Project Overview
We implemented the BDQ encoder consisting of three modules, Blur, Difference, and Quantization. The encoder is trained in an adversarial setup to retain action features while supressing privacy-sensitive ones. 

## Directory Structure 
### Core Components
- `bdq_encoder/` 
  - `BDQ.py`: high-level wrapper for the BDQ encoder
  - `blur.py`, `difference.py`, `quantization.py`: module-level definitions 

### Datasets & Metadata 
- Raw datasets available from: 
  - [KTH Dateset](https://www.csc.kth.se/cvap/actions/) 
  - [IXMAS Dataset](https://www.epfl.ch/labs/cvlab/data/data-ixmas10/) 
- `datasets/`
  - `datasets/KTH`: video files and `00sequences.txt` (metadata)
  - `datasets/IXAMS`: selected subset of IXMAS videos 
  - `datasets/util_kth.py`: action label map of the KTH dataset 
  - `datasets/ixmas_clips_6.json`, `datasets/kth_clips.json`: structured clip metadata generated from parsers 

### Notebooks 
- `notebooks/`: Ready-to-run experiments for the Kaggle GPU100 environment 
  - `notebooks/bdq-kth.ipynb`: KTH experiment with BDQ encoder 
  -  `notebooks/bdq-kth-no-encoder.ipynb`: KTH baseline without BDQ 
  -  `notebooks/bdq-ixmas.ipynb`: IXMAS experiment with BDQ encoder 
  -  `notebooks/bdq-ixmas-no-encoder.ipynb`: IXMAS baseline without BDQ 

### Preprocessing Scripts 
- `preprocess.py`: preprocessing pipeline 
- `raw_dataset_preprocess`: dataset-specific preprocessing tools 
  - `raw_dataset_preprocess/KTH_preprocess`
    - `.../KTH_preprocess/kth_parser.py`: parses `00sequences.txt` into structured KTH clip metadata 
  - `raw_dataset_preprocess/IXMAS_preprocess` 
    - `.../IXMAS_preprocess/IXMAS_720`: 720 manually selected representative frames 
    - `.../IXMAS_preprocess/IXMAS_utils` 
      - `.../ixmas_extract_frame.py`: extracts and saves a representative frame from each IXMAS video 
      - `.../IXMAS_utils/ixmas_extract_vid.py`: locates and copies videos matching selected frame names (e.g., for `IXMAS_720/`) 
      - `.../IXMAS_utils/ixmas_extract_vid_class.py`: filters and saves videos belonging to six selected actions 
      - `.../IXMAS_utils/ixmas_parser.py`: generates clip metadata from IXMAS video filenames 

### Models & Training 
- `loss.py`: loss functions for adversarial training 
- `training.py`: training and validation pipeline 
- `action_recognition_model.py`: 3D ResNet-based action classifier 
- `privacy_attribute_prediction_model.py`: 2D ResNet-based identity classifier 

### Visualization & Evaluation 
> todo

## Run the Project
1. Install requirements: 
```bash
pip install -r requirements.txt
```
2. Adversarial training: 
```bash
python training.py 
```

## Reproducing Figure 3 
> todo 

## References 
[1] S. Kumawat and H. Nagahara, “Privacy-Preserving Action Recognition via Motion Difference Quantization,” Aug. 2022.

[2] C. Schuldt, I. Laptev, and B. Caputo, “Recognizing human actions: A local SVM approach,” in Proceedings of the 17th International Conference on Pattern Recognition, 2004. ICPR 2004., (Cambridge, UK), pp. 32–36 Vol.3, IEEE, 2004.

[3] D. Weinland, R. Ronfard, and E. Boyer, “Free viewpoint action recognition using motion history volumes,” Computer Vision and Image Understanding, vol. 104, pp. 249–257, Nov. 2006. 
