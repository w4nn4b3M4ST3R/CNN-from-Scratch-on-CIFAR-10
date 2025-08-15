# CIFAR-10 CNN Classifier with PyTorch

![CNN Banner](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=PyTorch&logoColor=white)
![Python Badge](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=Python&logoColor=white)
![License Badge](https://img.shields.io/badge/License-MIT-green?style=flat)

This repository contains a complete implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset using PyTorch. The project includes data preparation, model definition, training, evaluation, and visualization scripts. It's designed for educational purposes and serves as a starting point for experimenting with CNNs in computer vision tasks.

## 📖 Overview

The project implements a custom CNN architecture (`MyNet`) to classify images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes (e.g., airplane, automobile, bird). Key features include:

- **Data Augmentation and Normalization**: Using transformations like random flips, crops, and normalization for better generalization.
- **Model Training**: With SGD optimizer, cross-entropy loss, and a learning rate scheduler.
- **Evaluation**: Metrics for training, validation, and test sets.
- **Visualization**: Plots for loss and accuracy over epochs.

This setup achieves around 75-85% accuracy on the test set after 50 epochs, depending on hardware and hyperparameters.

## 🔍 Theoretical Background

### CIFAR-10 Dataset
CIFAR-10 is a benchmark dataset for image classification, containing 50,000 training images and 10,000 test images across 10 classes. Each image is a 32x32 RGB pixel grid. The dataset challenges models due to its small size and variability in lighting, poses, and backgrounds.

### Convolutional Neural Networks (CNNs)
CNNs are specialized neural networks for processing grid-like data (e.g., images). They use:
- **Convolutional Layers**: To extract features like edges and textures using learnable filters.
- **Pooling Layers**: To reduce spatial dimensions and computational load (e.g., MaxPooling).
- **Activation Functions**: ReLU for non-linearity.
- **Batch Normalization**: To stabilize training by normalizing activations.
- **Dropout**: To prevent overfitting by randomly dropping neurons during training.

The loss function used is Cross-Entropy Loss, suitable for multi-class classification. Optimization is handled by Stochastic Gradient Descent (SGD) with momentum and weight decay for regularization.

### Data Normalization
Normalization subtracts the dataset mean and divides by the standard deviation per channel (R, G, B). For CIFAR-10, mean ≈ [0.4914, 0.4822, 0.4465] and std ≈ [0.2023, 0.1994, 0.2010]. This centers the data around zero, speeding up convergence and improving accuracy.

## 🏗️ Model Architecture

The custom CNN (`MyNet`) is defined in `model.py` and consists of sequential convolutional blocks followed by pooling and a classifier head.

### Key Components
- **Conv Block**: Conv2D (3x3 kernel, padding=1) → BatchNorm2D → ReLU.
- **Network Structure**:
  - Input: 3x32x32 (RGB image).
  - Conv Block (3 → 64) → Conv Block (64 → 64) → Conv Block (64 → 128) → MaxPool2D(2).
  - Conv Block (128 → 128) → Conv Block (128 → 256) → MaxPool2D(2).
  - Conv Block (256 → 256) → Conv Block (256 → 256) → AdaptiveAvgPool2D(1) → Flatten.
  - Dropout(0.3) → Linear(256 → 10 classes).
- **Parameters**: Approximately 1-2 million (viewable via `torchinfo.summary`).
- **Initialization**: Kaiming for Conv2D, Xavier for Linear layers.

This architecture draws inspiration from VGG-like networks but is simplified for efficiency.

```mermaid
graph TD
    A[Input: 3x32x32] --> B[Conv Block: 3->64]
    B --> C[Conv Block: 64->64]
    C --> D[Conv Block: 64->128]
    D --> E[MaxPool2D(2)]
    E --> F[Conv Block: 128->128]
    F --> G[Conv Block: 128->256]
    G --> H[MaxPool2D(2)]
    H --> I[Conv Block: 256->256]
    I --> J[Conv Block: 256->256]
    J --> K[AdaptiveAvgPool2D(1)]
    K --> L[Flatten]
    L --> M[Dropout(0.3)]
    M --> N[Linear: 256->10]
    N --> O[Output: Class Probabilities]
```

## ⚙️ Operating Principles

### Training Process
1. **Forward Pass**: Input images pass through the CNN to produce logits.
2. **Loss Calculation**: Cross-Entropy between predictions and labels.
3. **Backward Pass**: Gradients computed via backpropagation.
4. **Optimization**: SGD updates weights; scheduler reduces LR on validation accuracy plateau.
5. **Evaluation**: Accuracy and loss computed on validation set after each epoch.

### Inference
In test mode (`torch.inference_mode()`), the model evaluates without gradients, computing accuracy as the ratio of correct predictions to total samples.

### Overfitting Prevention
- Data augmentation (flips, crops).
- Dropout and weight decay.
- Early stopping potential via monitoring validation metrics.

## 🛠️ Implementation Details

### File Structure
- `prepare_data.py`: Loads CIFAR-10, applies transforms, splits into train/val/test loaders.
- `model.py`: Defines `MyNet` and conv blocks.
- `train.py`: Training loop with tqdm progress, metrics tracking.
- `test.py`: Evaluation on test set.
- `visualize.py`: Plots results using Matplotlib.
- `main.py`: Orchestrates data prep, model init, training, and visualization.

### Dependencies
- PyTorch (`torch`, `torchvision`)
- NumPy, Matplotlib, Scikit-learn, Tqdm, Torchinfo

Install via:
```bash
pip install torch torchvision torchinfo tqdm matplotlib scikit-learn
```

### Hyperparameters
- **Batch Size**: 128
- **Learning Rate**: 0.01 (with ReduceLROnPlateau scheduler, factor=0.5, patience=5)
- **Epochs**: 50 (adjustable, consider early stopping with patience=10)
- **Dropout**: 0.3
- **Weight Decay**: 0.001

## 🚀 Installation and Usage

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (optional, fallback to CPU if unavailable)
- Disk space for CIFAR-10 (~170 MB)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cifar10-cnn-pytorch.git
   cd cifar10-cnn-pytorch
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Create `requirements.txt` with:
   ```
   torch
   torchvision
   torchinfo
   tqdm
   matplotlib
   scikit-learn
   ```

3. Verify CUDA (if using GPU):
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

### Usage
Run the main script:
```bash
python main.py
```
- Downloads CIFAR-10 to `../dataset/` if not present.
- Trains for 50 epochs, displays model summary, and plots results.
- To evaluate on test set, add to `main.py`:
  ```python
  from test import test
  test_results = test(model, loss_func, test_loader, device)
  print(f"Test Loss: {test_results['test_loss']:.4f}, Test Accuracy: {test_results['test_acc']:.4f}")
  ```

Customize:
- Adjust `epochs` in `main.py`.
- Implement early stopping in `train.py` (see below).
- Save model: `torch.save(model.state_dict(), "mynet.pth")`.

### Early Stopping (Optional)
Add to `train.py` to stop training if validation accuracy plateaus:
```python
best_val_acc = 0.0
patience = 10
patience_counter = 0
for epoch in range(epochs):
    # ... (existing training code) ...
    if results["val_acc"][-1] > best_val_acc:
        best_val_acc = results["val_acc"][-1]
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"[INFO] Early stopping at epoch {epoch+1}")
        break
```

## 📊 Expected Results

After 50 epochs on a standard GPU:
- **Train Accuracy**: ~85%
- **Validation Accuracy**: ~80%
- **Test Accuracy**: ~78-82% (after adding test call)

Example Visualization (from `visualize.py`):
- Loss Plot: Train and validation loss over epochs.
- Accuracy Plot: Train and validation accuracy over epochs.

To save plots:
```python
plt.savefig("results.png")
```

## 🔧 Improvements and Future Work
- **Enhanced Data Augmentation**: Add `ColorJitter` or `RandomRotation`.
- **Advanced Architectures**: Experiment with ResNet or EfficientNet for >90% accuracy.
- **Hyperparameter Tuning**: Grid search for learning rate, batch size, etc.
- **Interactive Visualization**: Use Chart.js for web-based plots (contact for implementation).

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **Star this repo** if you find it useful! For questions or issues, please open an issue on GitHub.