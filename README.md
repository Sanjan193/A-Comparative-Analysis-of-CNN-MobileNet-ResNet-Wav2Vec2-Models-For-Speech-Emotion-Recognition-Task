# Speech Emotion Recognition: Comparative Study of Wav2Vec2, ResNet-52, MobileNetV3 and Custom ResNet

## 1. Project Overview

This repository presents a **controlled comparative study** of four deep learning architectures for **Speech Emotion Recognition (SER)** using PyTorch:

- **Wav2Vec2** (Meta)
- **ResNet-52**
- **MobileNetV3**
- **Custom ResNet-type architecture** (designed by me)

All models were trained and evaluated under the same conditions (same dataset split, same preprocessing pipeline, and same evaluation metrics) to ensure a fair comparison.

---

## 2. Comparative Study

The goal of this study is to analyze how different architectural families perform on the speech emotion recognition task:

| Model              | Type                        | Key Characteristic                     |
|--------------------|-----------------------------|----------------------------------------|
| Wav2Vec2           | Self-supervised Transformer | Pretrained on large audio data         |
| ResNet-52          | Residual CNN                | Deep residual connections              |
| MobileNetV3        | Efficient CNN               | Lightweight & mobile-friendly          |
| Custom ResNet      | Custom Residual Network     | Designed specifically for this task    |

Each model was trained in **two phases**:

- **Phase 1**: Initial training
- **Phase 2**: Further fine-tuning / continued training

---

## 3. Dataset

- **Dataset used**: [Write dataset name here, e.g. RAVDESS / IEMOCAP / CREMA-D / TESS etc.]
- **Emotions**: [List the emotions, e.g. Angry, Happy, Sad, Neutral, Fear, Disgust, Surprise]
- **Total samples**: 
- **Train / Validation / Test split**: 
- **Sampling rate**: 16 kHz
- **Input**: [Raw waveform for Wav2Vec2 | Mel-spectrogram for CNNs]

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
