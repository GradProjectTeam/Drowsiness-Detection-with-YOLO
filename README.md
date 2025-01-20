# Drowsiness Detection Using Computer Vision with YOLOv11n

## Project Overview

This project focuses on developing a **drowsiness detection system** using computer vision techniques. The system leverages a pre-trained **YOLOv11n** model, fine-tuned to detect states of alertness, including **drowsiness**, **awake**, and **yawning**. YOLOv11n, a variant of the YOLO (You Only Look Once) family of object detection models, is known for its efficiency and precision, making it ideal for real-time applications such as driver monitoring systems or workplace fatigue detection.

## Key Features

- **Real-Time Detection**: The system is designed to detect drowsiness, wakefulness, and yawning in real-time, making it suitable for safety-critical environments.
- **High Precision and Recall**: The model achieves high precision and recall across all classes, ensuring accurate detection of drowsiness and other states.
- **Fast Inference Speed**: With an inference speed of **1.9ms per image**, the model is optimized for real-time applications.
- **Custom Dataset**: The model is fine-tuned on a custom dataset specifically designed for drowsiness detection, ensuring robust performance in real-world scenarios.

## Dataset

The model is pre-trained on the **COCO dataset** and fine-tuned using a **custom dataset** for drowsiness detection. The custom dataset consists of images classified into three categories:

1. **Drowsiness**: Images of individuals showing signs of tiredness or drowsiness.
2. **Awake**: Images of individuals who are alert and awake.
3. **Yawn**: Images of individuals caught in the act of yawning.

The custom dataset is available on Kaggle: [Drowsiness Detection for YOLOv8](https://www.kaggle.com/datasets/cubeai/drowsiness-detection-for-yolov8).

## Model Architecture

The YOLOv11n model consists of **88 convolutional layers** distributed across the **backbone**, **neck**, and **detection head**:

- **Backbone**: 63 convolutional layers for feature extraction.
- **Neck**: 0 convolutional layers (uses upsampling and concatenation).
- **Detection Head**: 25 convolutional layers for predicting bounding boxes, objectness scores, and class probabilities.

The model uses **1x1 convolutional layers** instead of fully connected layers, which reduces computational complexity and preserves spatial hierarchy, enabling real-time detection.

## Training Details

The YOLOv11n model was fine-tuned on the custom dataset with the following hyperparameters:

- **Batch Size**: 128
- **Epochs**: 100
- **Learning Rate**: "auto" (adapted during training)
- **Freeze**: 11 (first 120 layers frozen)
- **Patience**: 10
- **Dropout**: 0.1

Data augmentation techniques such as flipping, rotation, and scaling were applied to enhance the model's robustness.

## Evaluation Metrics

The model was evaluated on a test dataset, and the following results were obtained:

- **Overall Results**:
  - **Precision (P)**: 0.963
  - **Recall (R)**: 0.94
  - **mAP50**: 0.984
  - **mAP50-95**: 0.87

- **Inference Speed**:
  - **Preprocess**: 0.2ms per image
  - **Inference**: 1.9ms per image
  - **Postprocess**: 1.2ms per image

## Future Improvements

1. **Fine-tune Hyperparameters**: Experimenting with different learning rates, batch sizes, and epochs could further enhance performance, especially for the yawning class.
2. **Increase Data Diversity**: Adding more diverse examples of yawning and drowsiness, including variations in lighting and posture, could improve the model's ability to generalize.
3. **Ensemble Approaches**: Combining multiple models or employing advanced techniques such as transfer learning might boost recall and precision for harder-to-detect cases, such as yawning.

## Usage

To use this model for drowsiness detection, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/abdoghazala7/Drowsiness-Detection
   cd Drowsiness-Detection

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   
3. **Run Inference**:
   ```bash
   python detect.py --source <path_to_image_or_video> --weights best.pt
   
4. **Real-Time Testing**:
   ```bash
   python test_on_reallive.py --weights best.pt
