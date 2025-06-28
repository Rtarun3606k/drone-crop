# Binary Neural Network - Plant Disease Classification

## Enhanced Testing Features

Your Binary Neural Network now includes comprehensive testing capabilities with adjustable sample sizes and full dataset evaluation.

### Available Test Functions

#### 1. Random Testing with Adjustable Sample Size

**Quick Test** (100 samples per class):

```python
from predectorAllimges import quick_test
result = quick_test()
```

**Standard Test** (1000 samples per class):

```python
from predectorAllimges import standard_test
result = standard_test()
```

**Large Test** (2000 samples per class):

```python
from predectorAllimges import large_test
result = large_test()
```

**Custom Test** (specify your own sample size):

```python
from predectorAllimges import custom_test
result = custom_test(1500)  # 1500 samples per class
```

#### 2. Comprehensive Testing (ALL Images)

Test every single image in your dataset:

```python
from predectorAllimges import complete_test
result = complete_test()
```

### Easy Command Line Interface

Use the `test_bnn.py` script for easy command-line testing:

```bash
# Quick test (100 samples per class)
python test_bnn.py quick

# Standard test (1000 samples per class)
python test_bnn.py standard

# Large test (2000 samples per class)
python test_bnn.py large

# Custom test with specified sample size
python test_bnn.py custom 1500

# Test ALL images in dataset
python test_bnn.py complete

# Interactive mode
python test_bnn.py
```

### Advanced Usage

For more control, use the main functions directly:

```python
from predectorAllimges import run_random_prediction_test, run_comprehensive_test

# Random test with custom sample size
result = run_random_prediction_test(sample_size=2500)

# Comprehensive test on all images
result = run_comprehensive_test()
```

### Output Files

All tests generate comprehensive results:

**Random Tests (`random_test_results/` directory):**

- `random_prediction_results_TIMESTAMP.csv` - Detailed predictions
- `prediction_analysis_TIMESTAMP.png` - 12-panel analysis graphs
- `classification_report_TIMESTAMP.txt` - Performance metrics

**Comprehensive Tests (`comprehensive_test_results/` directory):**

- `comprehensive_test_results_TIMESTAMP.csv` - All image predictions
- `comprehensive_analysis_TIMESTAMP.png` - Complete analysis graphs
- `comprehensive_report_TIMESTAMP.txt` - Full performance report

### Sample Size Recommendations

- **Quick Test (100)**: Fast development testing
- **Standard Test (1000)**: Good balance of speed and reliability
- **Large Test (2000)**: More robust statistical analysis
- **Custom**: Adjust based on your specific needs
- **Complete**: Ultimate evaluation on entire dataset

### Key Features

✅ **Adjustable sample sizes** - Easy parameter control  
✅ **Comprehensive testing** - Evaluate entire dataset  
✅ **Rich visualizations** - 12-panel analysis graphs  
✅ **Detailed metrics** - Per-class performance stats  
✅ **CSV exports** - All results saved for analysis  
✅ **Progress tracking** - Real-time testing updates  
✅ **Error handling** - Robust execution  
✅ **Command line interface** - Easy script execution

### Dataset Structure

Your dataset is now configured to use the MH-SoyaHealthVision dataset:

```
/home/dragoon/Downloads/MH-SoyaHealthVision An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment/Soyabean_UAV-Based_Image_Dataset/
├── Soyabean_Mosaic/ (772 images)
├── rust/ (1000 images)
└── Healthy_Soyabean/ (280 images)
```

**Dataset Statistics:**

- **Soyabean_Mosaic**: 772 images
- **rust**: 1000 images
- **healthy**: 280 images
- **Total**: 2052 images

### Model Files

Tests automatically use the most recent model from:

```
results/bnn_plant_disease_model_*.pth
```

Make sure to train your model first using the Jupyter notebook before running tests!
