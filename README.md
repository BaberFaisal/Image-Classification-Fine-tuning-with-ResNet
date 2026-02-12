# Image Classification Fine-tuning with ResNet-18

This repository contains a comprehensive guide and implementation for **Transfer Learning** using the PyTorch ecosystem. It demonstrates how to leverage state-of-the-art pre-trained models to solve image classification tasks efficiently.

##  Project Overview

The core of this project is the `Finetuning.ipynb` notebook, which walks through:
- **Transfer Learning Concepts**: Understanding how weights pre-trained on ImageNet can be repurposed for new tasks.
- **ResNet Architecture**: A deep dive into the ResNet-18 structure, including residual blocks and skip connections.
- **Model Customization**: Preparing a pre-trained model for fine-tuning by identifying and modifying the classification head.
- **Inference Pipeline**: Loading class labels and interpreting raw model logits as human-readable categories.

##  Technical Stack

- **Framework**: [PyTorch](https://pytorch.org/) (v2.0+)
- **Domain Library**: [TorchVision](https://pytorch.org/vision/main/index.html)
- **Hardware**: CUDA-enabled GPU (recommended for training performance)
- **Utilities**: PIL (Pillow), Matplotlib, NumPy, TQDM

##  Dataset: ImageNet

The project leverages the **ImageNet-1K** benchmark:
- **Training Set**: ~1.28 million images.
- **Validation Set**: 50,000 images.
- **Scope**: 1,000 distinct object categories.

##  Model Logic

The implementation uses a pre-trained **ResNet-18** model. The key steps include:
1. **Model Loading**: Initializing ResNet-18 with `ResNet18_Weights.DEFAULT`.
2. **Feature Extraction**: Using convolutional layers to identify complex visual patterns.
3. **Logit Transformation**: Mapping the 512-feature vector output by the average pooling layer to the final class scores via the `fc` (Fully Connected) layer.

[Image of ResNet architecture diagram showing skip connections and layers]

##  Quick Start

### Installation
```bash
pip install torch torchvision numpy matplotlib pillow requests tqdm
```

### Usage Snippet
```python
from torchvision.models import resnet18, ResNet18_Weights

# Load the best available weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval() # Set to evaluation mode

# Inspect the final layer for modification
print(model.fc) 
# Output: Linear(in_features=512, out_features=1000, bias=True)
```

##  Credits
This project utilizes the ImageNet dataset and TorchVision pre-trained models. Special thanks to the research community at Princeton (Fei-Fei Li et al.) for the ImageNet initiative.
