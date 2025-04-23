# Custom Object Detection from Scratch

A small object detection pipeline built from scratch using PyTorch. The goal is to understand and implement the full workflow — from loading YOLO-format data to building the model, loss function, training, and evaluation.

## Project Structure:

```
/custom-object-detector
├── /config
│   └── config.py           # YAML files for model architecture
├── /data
│   ├── loader.py           # data loader (handle only yolo format)
│   └── augment.py
├── /model
│   ├── arch.py             # Model architecture builder
│   └── loss.py             # Detection loss
├── /utils
│   ├── utils.py            # NMS, decoding, model saving/loading
│   ├── metrics.py          # Evaluation metrics
│   ├── filter.py           # Dataset filtering
│   ├── visualize.py        # Visualization utility
│   └── count.py            # Class distribution counter
├── train.py
├── eval.py
├── inference.py
├── requirements.txt
├── .gitignore
├── README.md
```

## Dataset 
The [PPE_DATASET](https://www.kaggle.com/datasets/shlokraval/ppe-dataset-yolov8?resource=download) was used from Kaggle. Only two classes are used: ['Hardhat', 'NO-Hardhat']. Corrupted and empty images were removed (see the filtered_data folder for excluded filenames). The original train/val split is preserved.

## How to Run
### 1. Install Dependencies
<br>

```
sh install.sh
```
<br>

### 2. Download and Prepare the Dataset (or any other datasets in YOLO format)
<br>

#### Download via kagglehub (or visit the website)
<br>

```
pip install kagglehub

python -c "
import kagglehub
path = kagglehub.dataset_download('shlokraval/ppe-dataset-yolov8')
"
```
#### Clean the Dataset
<br>


[filter.py]('./utils/filter.py): Filter the labels based on choosen classes. Also, remove empty and corrupted files.

[visualize.py](./utils/visualize.py): Check bounding boxes and images.

[count.py](./utils/count.py): inspect class distribution

<br>

### 3. Train the Model
<br>

```
sh ./train.sh
```
<br>

### 4. Evaluate the Model
<br>

```
sh ./evaluate.sh
```


# Plan
The development approach is divided into two main phases:

## Phase 1: Build a Complete Basic Pipeline
* Implement a simple but complete object detection pipeline

* Create modular building blocks for each stage:

    -  Data loading (supports YOLO format only)

    -  Model builder

    - Model training

    - Evaluation

    - Inference

* Test each component independently to ensure correctness

## Phase 2: Incremental Enhancements
* Gradually increase the complexity of the model architecture

* Integrate more advanced data augmentation techniques

* Optimize preprocessing and postprocessing functions (fully vectorized for speed)

## Architectures Used:

*  The first version is a custom CNN built from scratch with a simple sequential architecture. It uses a series of convolutional layers followed by ReLU activations and max pooling. The detection head predicts one bounding box and class scores per grid cell without using anchors.

* The second version used ResNet18 as a backbone connected to a custom detection head (same as the first version).

You can check the [configs](./configs) folder to explore the model architecture YAML files.

## Current State:
There's an issue in the postprocessing stage after training — specifically with decoding predictions. This is still under debugging. So no evaluation results to present.

