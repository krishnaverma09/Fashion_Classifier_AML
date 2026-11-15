---
title: Clothing Category Classifier
emoji: 👕
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: ResNet classifies clothing images into 10 categories
---

# Clothing Category Classifier

Upload a picture of a clothing item and the model predicts its category: T-shirt/top,
Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag or Ankle boot.

**Model:** a residual CNN (ResNet, 696,042 parameters) trained on 48,000 Fashion-MNIST
images with flip, shift and random-erasing augmentation. **Test accuracy: 94.32%**
(macro F1 94.31%), up from 85.27% for the best classical model (Random Forest).

## How to use

1. Upload an image or pick a sample.
2. Choose whether the background is dark or light.
3. Click **Predict category**.

## Limitations

Trained on 28×28 grayscale thumbnails. Real photos work best with one upright item
on a plain background. Scores are softmax values, not calibrated confidence.

## Links

- Code and report: https://github.com/krishnaverma09/Fashion_Classifier_AML
- Team: Satyam Kumar, Krishna Verma, Akshit Vats, Chinmay Soni, Lakshya Bapna
  (Newton School of Technology)
