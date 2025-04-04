
# Logo Detection with YOLOv8

This project aims to extract logos (patterns) from clothing, hats, socks, etc. using YOLOv8 for object detection. The dataset is a subset of the [LogoDet-3K Dataset](https://github.com/Wangjing1551/LogoDet-3K-Dataset?tab=readme-ov-file).

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/qianyiyang1/yolo8-logo-detection.git
   cd yolo8-logo-detection
   ```

2. **Install dependencies:**

   You will need Python 3.8+ and the required libraries. Install them using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure that you have the `ultralytics` library installed for YOLOv8 support.

3. **Dataset:**

   The dataset used in this project is a portion of the [LogoDet-3K Dataset](https://github.com/Wangjing1551/LogoDet-3K-Dataset?tab=readme-ov-file). You can visualize the dataset using the provided `visualize_gt.py` script.

4. **Pre-trained Model:**

   A pre-trained model (`best_0404.pt`) is available for direct use. You can use this for prediction without the need for additional training.

## Scripts

### 1. `visualize_gt.py`

This script helps visualize the ground truth annotations on the images. In the original dataset, logos were often annotated separately from the background patterns (for example, a "Celine" logo might be separated from the background design). However, in this project, we want to treat both the logo and the pattern as a single object. This script will help  you pick the right images for your model.

#### How to use:

```bash
python visualize_gt.py
```

It will create visualized images with bounding boxes based on the annotations and save them in the output directory.

### 2. `voc2yolo.py`

This script converts VOC format annotations to YOLO format, which is required for YOLOv8 training.

#### How to use:

```bash
python voc2yolo.py
```

This will convert the annotations and save them in the YOLO format.

### 3. `train.py`

This script is used to train the YOLOv8 model on the dataset. Make sure you have the dataset properly organized.

#### How to use:

```bash
python train.py
```

The training will use the YOLOv8 architecture with the provided dataset configuration (`dataset.yaml`). Other training parameters, such as the number of epochs, batch size, learning rate, etc., can be adjusted directly in the train.py script.

### 4. `predict.py`

This script allows you to perform inference using a pre-trained model.

#### How to use:

```bash
python predict.py
```

The script will perform inference and save the results.

## Dataset Structure

The dataset directory should be structured as follows:

```
data/dataset/
├── images/
│   ├── train/
│   └── val/
├── annotations/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```

- **images/train**: Contains the training images.
- **images/val**: Contains the validation images.
- **annotations/train**: Contains the training annotations in VOC format (XML).
- **annotations/val**: Contains the validation annotations in VOC format (XML).
- **labels/train**: Contains the YOLO-formatted label files for training.
- **labels/val**: Contains the YOLO-formatted label files for validation.

## Configurations

The dataset is configured in `dataset.yaml`:

```yaml
path: data/dataset/
train: images/train
val: images/val
nc: 1  # Modify according to the actual number of categories
names: ['logo']
```

## Pre-trained Model

You can use the provided pre-trained model (`best_0404.pt`) directly for prediction without needing to train the model again. Just run the `predict.py` script after setting the correct path.

## Requirements

- Python 3.8+
- `ultralytics` library for YOLOv8
- `opencv-python` for image processing
- `PyTorch` and `torchvision`
