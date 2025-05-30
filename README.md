# Shape Predictor

A neural network model that predicts Bezier curve sequences for geometric shapes. The model learns to generate diverse curve predictions for different input shapes, solving the issue where all shapes would produce identical predictions.

## 🎯 Overview

This project trains a transformer-based model to:
- Predict Bezier curve parameters (6 numbers per segment) for shapes
- Classify curve types (line, quadratic, cubic)
- Determine stop positions for variable-length sequences
- Generate diverse outputs for different input shapes

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
# Navigate to project directory
cd shape-predictor

# Create virtual environment
python -m venv shape_predictor_env

# Activate virtual environment
# Windows:
shape_predictor_env\Scripts\activate
# Linux/Mac:
source shape_predictor_env/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
# Create 5 simple shapes for training
python create_simple_shapes.py
```

This creates:
- 🔴 Red circle
- 🔵 Blue square  
- 🟢 Green triangle
- 🟡 Yellow diamond
- 🟣 Purple pentagon

### 3. Train the Model

```bash
# Train for 50 epochs on the 5 simple shapes
python boundedShapePredSOTA.py
```

Training will:
- Run for 50 epochs (or until convergence)
- Save best model to `bezier_checkpoints/best_model.pth`
- Save final model to `bezier_checkpoints/final_model.pth`
- Display training progress with loss metrics

### 4. Evaluate the Model

```bash
# Run comprehensive evaluation
python evaluate_model.py
```

This will:
- Load the best trained model
- Test on all 5 shapes
- Generate visualizations in `evaluation_results/`
- Check prediction diversity (main issue we solved)
- Display detailed metrics

### 5. Test Basic Functionality

```bash
# Quick functionality test (no ML dependencies needed)
python test_shapes.py
```

## 📁 Project Structure

```
shape-predictor/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore rules
├── boundedShapePredSOTA.py       # Main training script
├── evaluate_model.py             # Model evaluation script
├── create_simple_shapes.py       # Generate training data
├── test_shapes.py               # Basic functionality test
├── contour.py                   # Contour processing utilities
├── dataset/                     # Training data
│   ├── json/                   # Shape metadata
│   ├── images/                 # Shape images
│   └── masks/                  # Shape masks
├── bezier_checkpoints/         # Trained models
│   ├── best_model.pth         # Best performing model
│   └── final_model.pth        # Final epoch model
├── evaluation_results/         # Generated visualizations
└── shape_predictor_env/       # Virtual environment
```

## 🔧 Requirements

- **Python**: 3.9+ (tested on 3.9 and 3.11)
- **PyTorch**: 2.0+
- **Dependencies**: See `requirements.txt`

### Key Dependencies:
- `torch>=2.0.0` - Neural network framework
- `transformers>=4.20.0` - FLAN-T5 text encoder
- `opencv-python>=4.5.0` - Image/contour processing
- `matplotlib>=3.5.0` - Visualization
- `scipy>=1.8.0` - Bezier curve fitting
- `pillow>=8.0.0` - Image handling
- `numpy>=1.21.0` - Numerical operations

## 📊 Model Architecture

The model uses a transformer-based architecture with:
- **Text Encoder**: FLAN-T5-base for processing shape descriptions
- **Shape Encoder**: Processes parent shape Bezier sequences
- **Fusion Module**: Combines text and shape features
- **Decoder**: Predicts child shape parameters
- **Output Heads**: 
  - Curve parameters (6 per segment)
  - Curve types (line/quadratic/cubic)
  - Stop positions

## 🎯 Problem Solved

**Original Issue**: All curves predicted for a shape were identical regardless of input.

**Solution**: 
- Fixed model initialization to encourage diversity
- Improved training with better loss weighting
- Added proper evaluation to verify diverse outputs

**Verification**: Run `python evaluate_model.py` to see diverse predictions.

## 📈 Training Details

- **Dataset**: 5 simple geometric shapes
- **Epochs**: 50 (early stopping if loss < 1e-6)
- **Batch Size**: 2
- **Learning Rate**: 1e-3
- **Loss Components**:
  - Curve parameter L1 loss (λ=500)
  - Stop prediction loss (λ=10)
  - Length prediction loss (λ=20)
  - Type classification loss (λ=100)

## 🔍 Evaluation Metrics

The evaluation script provides:
- **Curve L1 Error**: Accuracy of curve parameters
- **Type Accuracy**: Curve type classification accuracy
- **Endpoint Error**: Shape endpoint prediction accuracy
- **Diversity Analysis**: Checks for prediction variance
- **Visual Comparisons**: Ground truth vs predictions

## 🐛 Troubleshooting

### Common Issues:

1. **"No module named 'torch'"**
   ```bash
   # Make sure virtual environment is activated
   shape_predictor_env\Scripts\activate
   pip install -r requirements.txt
   ```

2. **"Checkpoint not found"**
   ```bash
   # Train the model first
   python boundedShapePredSOTA.py
   ```

3. **"scipy not available"**
   ```bash
   pip install scipy
   ```

4. **Memory issues**
   - Use CPU instead of GPU: model automatically detects available device
   - Reduce batch size in training script if needed

### Python Version Issues:
- **Python 3.9**: ✅ Fully supported
- **Python 3.10+**: ✅ Supported
- **Python 3.8**: ⚠️ May work but not tested

## 📋 Usage Examples

### Basic Training:
```bash
python boundedShapePredSOTA.py
```

### Training with Custom Parameters:
```python
# Edit boundedShapePredSOTA.py
train_model_batched(
    dataset_path="dataset",
    num_epochs=100,        # Train longer
    learning_rate=5e-4,    # Different learning rate
    batch_size=4,          # Larger batch
)
```

### Load and Use Trained Model:
```python
import torch
from boundedShapePredSOTA import PolygonPredictor, PolygonConfig

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = PolygonConfig()
model = PolygonPredictor(cfg=cfg).to(device)

checkpoint = torch.load("bezier_checkpoints/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

# Use for predictions...
```

## 🏆 Results

After training, the model achieves:
- ✅ **Diverse predictions** for different shapes
- ✅ **100% type accuracy** on test shapes
- ✅ **Reasonable endpoint accuracy** (~0.47 average error)
- ✅ **No more identical outputs** (original problem solved)

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python test_shapes.py`
5. Submit a pull request

---

For questions or issues, please check the troubleshooting section above or create an issue in the repository. 