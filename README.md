# A Comparative Study of Wav2Vec2, ResNet-52, MobileNetV3 and Custom AlexNet Models for Speech Emotion Recognition Tasks On Voice Data 

## 1. Project Overview

This repository presents a **Comparative study** of four deep learning architectures for **Speech Emotion Recognition (SER)** by a PyTorch based apporach:

- **Wav2Vec2** (By Meta)
- **ResNet-512**
- **MobileNetV3 Large**
- **Custom ResNet-type architecture** (designed by me)

All models were trained and evaluated under these conditions (same dataset split, same preprocessing pipeline, and same evaluation metrics) to ensure a fair comparison.

---

## 2. Comparative Study

The goal of this study is to analyze how different architectural families perform on the speech emotion recognition task:

| Model              | Type                        | 
|--------------------|-----------------------------|
| Wav2Vec2           | Self-supervised Transformer | 
| ResNet-52          | Residual CNN                | 
| MobileNetV3        | Efficient CNN               |
| Custom ResNet-152      | Custom Residual Network |

Each model was trained in **two phases**:

- **Phase 1**: Initial training
- **Phase 2**: Further fine-tuning / continued training

---

## 3. Dataset

- **Dataset used**: RAVDESS Emotional speech audio(Got this from the kaggle here is the link https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
- **Emotions**: In this dataset we have total 8 types of emotion classes(01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised) 
- **Total samples**: Total 1440 samples are present in t his dataset 
- **Train / Validation / Test split**: 70:15:15 ratio
- **Sampling rate**: 16 kHz
- **Input**: 1.Raw waveform is given for Wav2Vec2 model becasue its not trained on mel-spectogram heat map images

  2. Mel-spectrogram heatmap image data is given for CNNs

---

## 4. Results Table

### Phase 1 Results

| Model              | Accuracy | Precision | Recall | F1-Score | Train Loss | Val Loss |
|--------------------|----------|-----------|--------|----------|------------|----------|
| Wav2Vec2           |          |           |        |          |            |          |
| ResNet-52          |          |           |        |          |            |          |
| MobileNetV3        |          |           |        |          |            |          |
| Custom ResNet      |          |           |        |          |            |          |

### Phase 2 Results

| Model              | Accuracy | Precision | Recall | F1-Score | Train Loss | Val Loss |
|--------------------|----------|-----------|--------|----------|------------|----------|
| Wav2Vec2           |          |           |        |          |            |          |
| ResNet-52          |          |           |        |          |            |          |
| MobileNetV3        |          |           |        |          |            |          |
| Custom ResNet      |          |           |        |          |            |          |

---

## 5. Model Architectures & Training Curves

### 5.1 Wav2Vec2

![Wav2Vec2 Architecture](assets/architectures/wav2vec2_architecture.png)

**Phase 1 – Loss & Accuracy**  
![Phase 1](results/plots/wav2vec2/phase1_loss_acc.png)

**Phase 2 – Loss & Accuracy**  
![Phase 2](results/plots/wav2vec2/phase2_loss_acc.png)

---

### 5.2 ResNet-52

![ResNet-52 Architecture](assets/architectures/resnet52_architecture.png)

**Phase 1 – Loss & Accuracy**  
![Phase 1](results/plots/resnet52/phase1_loss_acc.png)

**Phase 2 – Loss & Accuracy**  
![Phase 2](results/plots/resnet52/phase2_loss_acc.png)

---

### 5.3 MobileNetV3

![MobileNetV3 Architecture](assets/architectures/mobilenetv3_architecture.png)

**Phase 1 – Loss & Accuracy**  
![Phase 1](results/plots/mobilenetv3/phase1_loss_acc.png)

**Phase 2 – Loss & Accuracy**  
![Phase 2](results/plots/mobilenetv3/phase2_loss_acc.png)

---

### 5.4 Custom ResNet-type Architecture

![Custom ResNet Architecture](assets/architectures/custom_resnet_architecture.png)

**Phase 1 – Loss & Accuracy**  
![Phase 1](results/plots/custom_resnet/phase1_loss_acc.png)

**Phase 2 – Loss & Accuracy**  
![Phase 2](results/plots/custom_resnet/phase2_loss_acc.png)

---

## 6. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model
python src/train.py --model wav2vec2 --phase 1
python src/train.py --model resnet52 --phase 2

# Evaluate
python src/evaluate.py --model wav2vec2 --phase 2
