# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Vani
**Team Members:** Sarita, Gargi   
**Solution Type:** Multi-Modal Machine Learning (Text + Image)  
**Primary Model:** LightGBM Regression with Feature Fusion

---

## 1. Executive Summary

This solution addresses the Smart Product Pricing challenge through a multi-modal machine learning approach that combines advanced text processing and comprehensive image analysis. Our system processes 50,000 training samples with catalog content and product images to predict optimal pricing using a LightGBM regression model.

**Key Achievements:**
- Multi-modal feature extraction combining text and image data
- Memory-optimized processing for large-scale datasets
- Flexible execution modes for different computational constraints
- Robust error handling with graceful fallbacks
- SMAPE scores ranging from 10-20% depending on execution mode

---

## 2. Methodology Overview

### 2.1 Problem Analysis

**Challenge:** Predict product prices from heterogeneous data (text + images) with complex relationships between product attributes and pricing.

**Key Insights:**
- Brand recognition significantly impacts pricing
- Product categories have distinct pricing patterns
- Visual features (color, texture, composition) correlate with price ranges
- Text semantic clustering reveals premium vs budget product segments
- Item Pack Quantity (IPQ) directly influences unit pricing

### 2.2 Solution Strategy

**Multi-Modal Approach:**
1. **Text Processing:** Extract semantic features, brand information, and pricing indicators
2. **Image Analysis:** Capture visual characteristics through traditional and deep learning features
3. **Feature Fusion:** Combine text and image features for holistic product representation
4. **Regression Modeling:** Use LightGBM for robust price prediction with log-transform

**Memory Optimization:**
- Chunked processing to handle large datasets
- Multiple execution modes for different hardware constraints
- Efficient feature storage and retrieval

---

## 3. Model Architecture

### 3.1 Architecture Overview

```
Input Data (Text + Images)
         ↓
┌─────────────────┬─────────────────┐
│   Text Pipeline │  Image Pipeline │
│                 │                 │
│ • TF-IDF        │ • Color Features│
│ • Brand Extract │ • Texture Anal. │
│ • Semantic Clust│ • Composition   │
│ • Premium Detect│ • Deep Features │
└─────────────────┴─────────────────┘
         ↓
    Feature Fusion
         ↓
    LightGBM Regressor
         ↓
    Price Prediction
```

### 3.2 Model Components

**Text Processing Engine:**
- TF-IDF Vectorization (5000 features)
- Brand extraction using regex patterns
- Semantic clustering with KMeans
- Premium product detection
- Statistical text features (length, word count)

**Image Processing Engine:**
- Traditional features: Color analysis, texture variance, composition metrics
- Deep features: ResNet50/EfficientNet embeddings (2048 dimensions)
- Error handling for missing/corrupted images

**Regression Model:**
- LightGBM with L1 objective (MAE optimization)
- Log-transform for price normalization
- Hyperparameters: 1000 estimators, 0.05 learning rate

---

## 4. Feature Engineering Innovations

### 4.1 Text Features

**Advanced Text Engineering (13 core features + 5000 TF-IDF):**

1. **Semantic Analysis:**
   - TF-IDF vectorization with 5000 features
   - Semantic clustering (5 clusters) using KMeans
   - Text length and word count statistics

2. **Brand Intelligence:**
   - Regex-based brand extraction from 200+ known brands
   - Brand encoding and premium brand detection
   - Brand-specific pricing patterns

3. **Premium Detection:**
   - Keyword-based premium product identification
   - Luxury indicators in product descriptions
   - Price range categorization

4. **Quantity Analysis:**
   - Item Pack Quantity (IPQ) extraction
   - Unit price normalization
   - Bulk pricing detection

### 4.2 Image Features

**Comprehensive Visual Analysis (13 traditional + 2048 deep features):**

1. **Color Analysis:**
   - Brightness and contrast metrics
   - Dominant color extraction (RGB values)
   - Color variance and richness measures

2. **Texture Features:**
   - Texture variance analysis
   - Edge density using Canny detection
   - Gradient magnitude computation

3. **Composition Metrics:**
   - Aspect ratio analysis
   - Center vs edge brightness contrast
   - Image complexity measures

4. **Deep Learning Features:**
   - ResNet50 embeddings (2048 dimensions)
   - EfficientNet alternative (1280 dimensions)
   - Transfer learning from ImageNet

---

## 5. Model Performance

### 5.1 Validation Results

**Performance by Execution Mode:**

| Mode | Dataset Size | Features | SMAPE Score | Processing Time |
|------|-------------|----------|-------------|----------------|
| **Local Mode** | 50,000 | Text + Deep Images | ~10-15% | 2-4 hours |
| **Light Mode** | 5,000 | Text + Traditional | ~15-20% | 30-60 min |
| **Cloud Mode** | 50,000 | Text + Deep Images | ~10-15% | 1-2 hours |

**Feature Importance Analysis:**
- Text features contribute ~60% to model performance
- Brand information is the strongest single predictor
- Deep image features improve accuracy by 2-3% over traditional
- Semantic clustering reduces noise in text representations

### 5.2 Key Performance Factors

**Success Factors:**
1. **Multi-modal fusion** captures both textual and visual pricing cues
2. **Brand extraction** identifies premium vs budget segments
3. **Log-transform** handles wide price range distribution
4. **Robust preprocessing** maintains performance despite missing data
5. **Memory optimization** enables processing of large datasets

**Limitations:**
- Deep learning features require significant computational resources
- Google Drive API has rate limits affecting processing speed
- Model performance depends on image quality and availability

---

## 6. Technical Implementation

### 6.1 Execution Modes

**Three optimized execution modes for different constraints:**

1. **Local Mode** (`train_model.py`):
   - Downloads images locally for fastest processing
   - Full dataset with deep learning features
   - Best accuracy but highest memory usage (16GB+)

2. **Light Mode** (`train_model_drive_light.py`):
   - Memory-optimized with chunked processing
   - Subset training (5k samples) with traditional features
   - Suitable for limited hardware (4-8GB RAM)

3. **Cloud Mode** (`train_model_drive.py`):
   - Direct Google Drive integration
   - Full dataset with comprehensive features
   - Designed for cloud computing environments

## 7. Conclusion

Our multi-modal approach successfully addresses the Smart Product Pricing challenge by combining advanced text processing with comprehensive image analysis. The solution demonstrates strong performance across different computational constraints while maintaining robustness through intelligent error handling and memory optimization.

**Key Innovations:**
- Flexible architecture supporting multiple execution modes
- Advanced feature engineering combining traditional and deep learning approaches
- Memory-efficient processing enabling large-scale dataset handling
- Robust preprocessing with graceful fallbacks for missing data

**Future Enhancements:**
- Integration of additional product metadata (reviews, ratings)
- Advanced ensemble methods combining multiple model architectures
- Real-time pricing optimization with market dynamics
- Enhanced image preprocessing for better feature extraction

## Appendix

### A. Code Structure

```
src/
├── text_features.py          # Advanced text processing pipeline
├── image_features.py         # Local image feature extraction
├── image_features_drive.py   # Google Drive image processing
└── drive_utils_fast.py       # Drive API utilities

Main Scripts:
├── train_model.py            # Local mode execution
├── train_model_drive.py      # Cloud mode execution
├── train_model_drive_light.py # Light mode execution
├── data_setup.py             # Image download utility
└── evaluate_results.py       # SMAPE evaluation

Output:
├── models/                   # Trained model components
├── dataset/test_out.csv      # Final predictions
└── images/                   # Downloaded images (local mode)
```

### B. Execution Workflow

**Training Phase:**
1. Load training data (50k samples)
2. Extract text features using TF-IDF and semantic analysis
3. Process images for visual features (traditional + deep)
4. Combine features and train LightGBM model
5. Save model components and metadata

**Prediction Phase:**
1. Load test data (10k samples)
2. Apply same feature extraction pipeline
3. Load trained model and generate predictions
4. Apply post-processing and create submission file

### C. Key Innovation: Separate Train/Predict Phases

**Modular Design Benefits:**
- **Flexibility:** Train once, predict multiple times
- **Efficiency:** Avoid retraining for new test data
- **Debugging:** Isolate training vs prediction issues
- **Scalability:** Deploy trained models independently
- **Memory Management:** Process large datasets in stages

**Component Persistence:**
- `trained_model.pkl`: LightGBM regression model
- `tfidf_vectorizer.pkl`: Fitted text vectorizer
- `feature_columns.pkl`: Feature metadata and column names
- `model_metadata.pkl`: Training statistics and configuration

---