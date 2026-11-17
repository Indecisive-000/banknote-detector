# Banknote Authenticity Detector

A small learning project: a convolutional neural network (TensorFlow/Keras) that classifies a photo of a banknote as `real` or `fake`.

## Status

Early prototype. The code for loading images, training and evaluating the model is written, but the dataset is still far too small to train a reliable model. No trained model or results are included yet.

## How it works

- Images are resized to 224×224 and pixel values are scaled to 0–1.
- The data is split 80/20 into training and test sets.
- The model is a simple CNN: three convolution + pooling layers, a dense layer with dropout, and a softmax output.
- After training, the script prints accuracy, F1-score, a classification report and a confusion matrix, then saves the model as `banknote_detector_model.h5`.
- `predict_single(model_path, image_path)` classifies a single photo.

## How to run

```
pip install -r requirements.txt
```

1. Put photos in `dataset/real/` and `dataset/fake/`.
2. In `banknote_detector.py`, uncomment `train_and_evaluate()` and run:
   ```
   python banknote_detector.py
   ```

## Next steps

- Collect a much larger dataset where both classes cover the same denominations and photo styles.
- Add data augmentation (rotation, brightness, noise).
- Try a pretrained model such as MobileNet instead of a CNN trained from scratch.
- Add a simple interface for uploading a photo.

---

This is a learning project, not a real tool for checking money.
