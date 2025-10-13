# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Vani  
**Team Members:** Sarita, Gargi
**Submission Date:** 14 October, 2025

---

## 1. Executive Summary

We developed a multimodal machine learning solution that combines advanced text and image feature extraction with LightGBM regression to predict product prices. Our approach leverages semantic clustering, premium packaging detection, and deep CNN features to achieve robust price prediction optimized for SMAPE metric.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

Product pricing depends on multiple factors including brand positioning, product quality, packaging aesthetics, and market segmentation. Our EDA revealed strong correlations between text richness, premium keywords, image quality indicators, and price ranges.

**Key Observations:**
- Premium keywords ("luxury", "gourmet", "organic") correlate with 2-3x higher prices
- Products with professional photography and gold/silver packaging command premium prices  
- Bulk indicators and pack sizes significantly impact total price
- Semantic product clustering reveals distinct price segments

### 2.2 Solution Strategy

**Approach Type:** Multimodal Feature Engineering + Gradient Boosting  
**Core Innovation:** Semantic product clustering combined with premium packaging detection from images, optimized for SMAPE metric through log-scale prediction

---

## 3. Model Architecture

### 3.1 Architecture Overview
```
Text Features (2000+ dims) -──┐
                              ├── Feature Fusion ──> LightGBM ──> Price Prediction
Image Features (2048+ dims) ──┘
```

### 3.2 Model Components

**Text Processing Pipeline:**
- Preprocessing: Enhanced cleaning, IPQ extraction, brand/category detection
- Semantic clustering: SBERT + MiniBatchKMeans (50 clusters)
- TF-IDF: 2000 features, sublinear scaling, bigrams
- Price indicators: Premium/budget keyword counting, bulk detection

**Image Processing Pipeline:**
- Traditional features: Color analysis, texture, composition (27 features)
- Premium packaging: Gold/silver/black color detection
- Deep features: ResNet50 pretrained on ImageNet (2048 features)
- Preprocessing: RGB conversion, 224x224 resize, ImageNet normalization

**Final Model:**
- Algorithm: LightGBM with L1 loss (MAE objective)
- Target transformation: log1p for better distribution
- Features: ~4000+ combined text and image features
- Hyperparameters: 1000 estimators, 0.05 learning rate, 31 leaves

---

## 4. Feature Engineering Innovations

### 4.1 Text Features
- **Semantic Clustering**: Groups similar products for price range estimation
- **Brand Extraction**: NER-based brand detection from product names
- **IPQ Standardization**: Unit conversion to base values (grams, milliliters)
- **Price Indicators**: Luxury vs budget keyword analysis

### 4.2 Image Features  
- **Premium Packaging Detection**: HSV-based gold/silver color analysis
- **Composition Analysis**: Center-focus and aspect ratio features
- **Quality Indicators**: Edge density and texture variance
- **Deep Visual Features**: ResNet50 transfer learning

---

## 5. Model Performance

### 5.1 Validation Results
- **Training SMAPE**: ~15-20% (varies by data split)
- **Feature Contribution**: Text features ~60%, Image features ~40%
- **Memory Efficiency**: Sparse matrix optimization, ~50MB total

### 5.2 Key Performance Factors
- Log-scale prediction reduces SMAPE for high-value products
- Semantic clustering provides strong price range priors
- Premium packaging detection captures luxury product premiums
- Multimodal fusion improves robustness over single-modality approaches

---

## 6. Technical Implementation

### 6.1 Execution Modes

**🚀 FAST MODE** (Development & Testing)
```bash
python train_model_drive_fast.py
```
- Text-only features, 2-5 minute training
- Uses: `models/fast_model.pkl`
- Best for: Quick iterations, development

**🎯 STANDARD MODE** (Production)
```bash
python data_setup.py
python train_model.py train
python train_model.py predict
```
- Multimodal features, 15-30 minute training
- Uses: `models/trained_model.pkl`
- Best for: Balanced accuracy vs speed

**⚡ ADVANCED MODE** (Competition)
```bash
python train_advanced.py
```
- Deep learning features, 45-90 minute training
- Best for: Maximum accuracy

**☁️ CLOUD MODE** (Large Scale)
```bash
python train_model_drive.py
```
- Google Drive integration for 75k+ images
- Memory-optimized batch processing

### 6.2 Libraries & Optimization

**Libraries Used:**
- Feature Engineering: scikit-learn, spaCy, sentence-transformers
- Image Processing: OpenCV, PIL, TensorFlow/Keras
- Modeling: LightGBM, scipy (sparse matrices)
- Data Processing: pandas, numpy

**Optimization Strategies:**
- Sparse matrix operations for memory efficiency
- Batch processing for image feature extraction
- Adaptive clustering for scalability
- SMAPE-optimized loss function (L1/MAE)
- Separate train/test workflows prevent data leakage

---

## 7. Conclusion

Our multimodal approach successfully combines textual product information with visual packaging cues to predict prices. The semantic clustering innovation and premium packaging detection provide significant improvements over baseline approaches. The solution is production-ready with efficient memory usage and robust error handling.

---

## Appendix

### A. Code Structure
```
Vani_Amazon_ML_Challenge_25/
├── dataset/
│   ├── train.csv, test.csv     # Main datasets (75k each)
│   └── test_out.csv           # Final submission file
├── src/
│   ├── text_features.py       # Advanced text processing
│   ├── image_features.py      # Comprehensive image features
│   ├── utils.py              # Image download utilities
│   └── drive_utils.py        # Google Drive integration
├── models/
│   ├── trained_model.pkl     # Standard mode model
│   ├── fast_model.pkl        # Fast mode model
│   └── tfidf_vectorizer.pkl  # Text transformers
├── images/
│   ├── train/                # Training images (75k)
│   └── test/                 # Test images (75k)
├── train_model.py            # Standard mode pipeline
├── train_model_drive_fast.py # Fast mode pipeline
├── train_advanced.py         # Advanced mode pipeline
├── train_model_drive.py      # Cloud mode pipeline
├── data_setup.py            # Image organization
└── workflow_example.py      # Automated workflow
```

### B. Execution Workflow
1. **Setup**: `source pricing_venv/bin/activate`
2. **Choose Mode**: Fast/Standard/Advanced/Cloud based on requirements
3. **Training**: Model learns from training data, saves components
4. **Prediction**: Loads saved model, predicts on test data
5. **Output**: `dataset/test_out.csv` ready for submission

### C. Key Innovation: Separate Train/Predict Phases
- **Training Phase**: Fits transformers only on training data
- **Prediction Phase**: Loads saved components, applies to test data
- **No Data Leakage**: Complete separation of training and test workflows
- **Production Ready**: Train once, predict multiple times

---

**Submission Ready:** All execution modes generate the required `test_out.csv` format optimized for SMAPE evaluation. Choose execution mode based on time constraints and accuracy requirements:

| Mode | Time | Accuracy | Best For |
|------|------|----------|----------|
| Fast | 2-5 min | Good | Development, testing |
| Standard | 15-30 min | Better | Production, balanced |
| Advanced | 45-90 min | Best | Competition submission |
| Cloud | Variable | Best | Large datasets, cloud processing |