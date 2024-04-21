# Human Detection in Nano Size with YOLOv9 👁️👁️

## Project Overview

This project focuses on training a YOLOv9 model for human detection in nano-sized images. The dataset used for training contains annotated images and labels specifically designed for detecting humans in small-scale images. The YOLOv9 model is trained to accurately identify and localize human objects within these nano-sized images.

## Dataset Creation

1. **Dataset Preparation**: Images and annotations were collected and organized specifically for human detection in nano-sized images.

2. **Annotation with Roboflow**: The dataset was annotated using Roboflow, a platform that provides tools for annotating images with bounding boxes for object detection tasks.

3. **Exporting Dataset**: The annotated dataset was exported in YOLOv8 format, compatible with YOLOv9 training.

## Setup and Requirements

1. **Environment**: Ensure you have a compatible environment with NVIDIA GPU support.
   - You can check GPU availability using `!nvidia-smi`.

2. **Mount Google Drive**: Mount your Google Drive to access files.
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Navigate to YOLOv9 Directory**:
    ```python
    %cd /content/drive/MyDrive/yolo9/yolov9
   ```
4. **Install Required Libraries**: YOLOv9 relies on PyTorch, OpenCV, and other libraries. Ensure they are installed.

## Training

## Command for Training:

```python
   !python train_dual.py --workers 8 --batch 4 --img 640 --epochs 10 --data /content/drive/MyDrive/yolo9/yolov9/data.yaml --weights /content/drive/MyDrive/yolo9/yolov9-e-converted.pt --device 0 --cfg /content/drive/MyDrive/yolo9/yolov9/models/detect/yolov9_custom.yaml --hyp /content/drive/MyDrive/yolo9/yolov9/data/hyps/hyp.scratch-high.yaml

   ```
--workers: Number of data loading workers.

--batch: Batch size for training.

--img: Input image size.

--epochs: Number of epochs to train.

--data: Path to data configuration file (.yaml).

--weights: Path to initial weights file.

--device: GPU device index (0 for first GPU).

--cfg: Path to model configuration file (.yaml).

--hyp: Path to hyperparameters file (.yaml).

## Detection

## Command for Detection:

```python
   !python detect.py --img 1080 --conf 0.1 --device 0 --weights /content/drive/MyDrive/yolo9/yolov9/runs/train/exp4/weights/best.pt --source /content/drive/MyDrive/yolo9/vi.video

   ```

--conf: Confidence threshold.

--device: GPU device index (0 for first GPU).

--weights: Path to trained weights file.

--source: Path to input image or video for detection.

## Additional Training and Detection

## You can repeat the training and detection process with different configurations or files by adjusting the command parameters accordingly.

## Feel free to modify and expand this README with more details or instructions based on your specific use case or project requirements.
