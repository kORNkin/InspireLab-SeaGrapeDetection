# Sea Grape Maturity Detection

### Example

Image             |  Model Detection
:-------------------------:|:-------------------------:
![](assets/raw_image.jpg)  |  ![](assets/detected_image.jpg)

## Navigation
- [Project Workflow](#project-workflow)
- [Development](#development)
    - [1. Prepare Data](#1-prepare-data)
    - [2. Process Data](#2-process-data)
    - [3. Build Dataset for Classifier](#3-build-dataset-for-classifier)
    - [4. Train the Detector](#4-train-the-detector)
    - [5. Train the Classifier](#5-train-the-classifier)
    - [6. Detect & Classify Sea Grapes](#6-detect--classify-sea-grapes)
- [Results](#results)


## Project Workflow

```mermaid
graph TD;
    image[Image] --> yolo("`**Stage 1**:<br>*YOLO11s* detects sea grapes`");
    yolo e1@--> |Sea grapes at all stages of maturity| classifier("`**Stage 2**:<br>*MobileNetV3* classifies maturity of each sea grape`");
    e1@{ animation: fast }
    classifier -- Combine detection and classification --> result[Predicted image with labels]
```

## Development

### 1. Prepare Data

**Problems**: The original dataset is missing many sea grape labels, which can confuse the model into treating real sea grapes as background.

**Method**: I trained a model on the original dataset and used it to pre-label the missing sea grapes, then manually reviewed and refined all annotations.

Original Data             |  Updated Data
:-------------------------:|:-------------------------:
![](assets/original_data.jpg)  |  ![](assets/updated_data.jpg)

---

### 2. Process Data 

> [pipeline/01_slice_tiles.py](pipeline/01_slice_tiles.py)

**Problems**: 
- The dataset contains 4640x3480 images, which are quite big but blurry. 
- Resizing them directly to YOLO's standard 640x640 input would shrink each grape (~80x79 px) down to roughly 11 px, making harder to detect.

**Method**: Each image is sliced into 640x640 tiles, matching YOLO's standard input size while preserving the grape's real pixel size.

---

### 3. Build Dataset for Classifier

> [pipeline/02_make_crops.py](pipeline/02_make_crops.py)

**Purpose**: Build a classifier dataset from YOLO dataset format.

**Methods**:
1. Extract each bounding box (one crop) with expanded 15% padding on each side from YOLO dataset
    > 15% padding is to show the classifier the neighboring grapes because maturity judgement is partly relative to the neighbors.
2. Split all bounding boxes (crops) into train/test/valid folders. 

---

### 4. Train the Detector

> [pipeline/03_train_detector.py](pipeline/03_train_detector.py)

**Purpose**: Train a detector model to detect sea grapes at all stages of maturity.

**Model**: YOLO11s — suitable model for our dataset size (few thousand tiles) and small object detection.

**Method**: Call `model.train()` with augmentation parameters.

---

### 5. Train the Classifier

> [pipeline/04_train_classifier.py](pipeline/04_train_classifier.py)

**Purpose**: Train a classifier model to classify maturity of each sea grape

**Model**: MobileNetV3-Small — a small and fast model to learn small dataset and remain fast classification.

**Method**: 
1. Load dataset from [the building dataset step](#3-build-dataset-for-classifier)
2. Create `torchvision.transforms` for training and evaluating the model
3. Balance class weight using `WeightedRandomSampler()` 
    > This addresses imbalanced dataset as the Whitening sample size is 13 times smaller than the Harvestable one.
4. Train the model 
5. After each epoch, evaluate macro recall on the validation set and save the model only when it improves
    > Macro recall averages recall per class, so the rare Whitening class counts as much as Harvestable. Plain accuracy would reward a model that always guesses Harvestable.

---

### 6. Detect & Classify Sea Grapes
> [pipeline/05_detect_classify.py](pipeline/05_detect_classify.py)

**Purpose**: Detect and classify sea grapes by combining the two stages.

**Methods**:
- Stage 1: `detect()`
    1. Slice input image into 640x640 tiles and stride 512px
        > Sliding window stride helps preventing the cut in half sea grapes.
    2. Create batches of 32 tiles and call `YOLO.predict()` for each
    3. Drop boxes that are likely the cut in half sea grapes in the batch
    4. Merges duplicated boxes from tile overlap
- Stage 2: `classify()`
    1. Load predicted boxes from Stage 1
    2. Expand each box by 15% padding
        > Following [the dataset](README.md?plain=65)

## Results
