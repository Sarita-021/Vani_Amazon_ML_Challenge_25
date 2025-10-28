# ML Challenge 2025 - Smart Product Pricing

## Smart Product Pricing Challenge

In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction. Your challenge is to develop an ML solution that analyzes product details and predict the price of the product. The relationship between product attributes and pricing is complex - with factors like brand, specifications, product quantity directly influence pricing. Your task is to build a model that can analyze these product details holistically and suggest an optimal price.

### Data Description:

The dataset consists of the following columns:

1. **sample_id:** A unique identifier for the input sample
2. **catalog_content:** Text field containing title, product description and an Item Pack Quantity(IPQ) concatenated.
3. **image_link:** Public URL where the product image is available for download. 
   Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg
   To download images use `download_images` function from `src/utils.py`. See sample code in `src/test.ipynb`.
4. **price:** Price of the product (Target variable - only available in training data)

### Dataset Details:

- **Training Dataset:** 50k products with complete product details and prices
- **Test Set:** 10k products for final evaluation

### Output Format:

The output file should be a CSV with 2 columns:

1. **sample_id:** The unique identifier of the data sample. Note the ID should match the test record sample_id.
2. **price:** A float value representing the predicted price of the product.

Note: Make sure to output a prediction for all sample IDs. If you have less/more number of output samples in the output file as compared to test.csv, your output won't be evaluated.

### File Descriptions:

*Source files*

1. **src/text_features.py:** Advanced text feature engineering with semantic clustering, brand extraction, and price indicators.
2. **src/image_features.py:** Comprehensive image feature extraction including color, texture, composition, and deep CNN features.
3. **src/image_features_drive.py:** Comprehensive image feature extraction using drive including color, texture, composition, and deep CNN features.
4. **src/drive_utils_fast.py:** Google Drive integration utilities for fast cloud-based processing.

*Dataset files*

1. **dataset/train.csv:** Training file with labels (`price`).
2. **dataset/test.csv:** Test file without output labels (`price`). Generate predictions using your model/solution on this file's data.
3. **dataset/test_out_correct.csv:** Test file with correct prices for SMAPE evaluation during development


### Evaluation Criteria:

Submissions are evaluated using **Symmetric Mean Absolute Percentage Error (SMAPE)**: A statistical measure that expresses the relative difference between predicted and actual values as a percentage, while treating positive and negative errors equally.

**Formula:**
```
SMAPE = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2)
```

**Example:** If actual price = $100 and predicted price = $120  
SMAPE = |100-120| / ((|100| + |120|)/2) * 100% = 18.18%

**Note:** SMAPE is bounded between 0% and 200%. Lower values indicate better performance.
---


## 🚀 ML Pipeline Execution Guide

### **Prerequisites / Setup**
```bash
# Create virtual environment 
python3 -m venv pricing_venv

# Activate virtual environment
source pricing_venv/bin/activate  # On macOS/Linux
# OR
pricing_venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Optional: For Google Drive integration
# Update config.py with your credentials.json and folder ID
```

### **Execution Modes by Complexity**


#### **🎯 LOCAL MODE (Images present locally)**
**Best for:** Production use, balanced accuracy vs speed
```bash
# 1. Download images to separate folders
python data_setup.py

# 2. Train model (saves components)
python train_model.py train

# 3. Generate predictions
python train_model.py predict

# 4. Evaluate SMAPE score (optional)
python evaluate_results.py
```
**Features:**
- Local image processing with OpenCV and PIL
- ResNet50/EfficientNet deep features
- Advanced text feature engineering
- Full dataset processing (50k samples)
- Highest accuracy potential

#### **⚡ LIGHT MODEL (Memory-Optimized)**
**Best for:** Limited memory systems, quick testing
```bash
# Memory-efficient pipeline
python train_model_drive_light.py full
```
**Features:**
- Traditional image features only (no deep learning)
- Chunked processing to prevent OOM
- Subset training (5k samples)
- Google Drive integration
- Fast execution

#### **☁️ CLOUD MODE (Google Drive Integration)**
**Best for:** Large datasets, cloud processing
```bash
# Cloud-optimized pipeline (use with caution - memory intensive)
python train_model_drive.py full
```
**Features:**
- Direct Google Drive image access
- Full deep learning pipeline
- Batch processing optimization
- Comprehensive feature extraction

### **Key Features & Innovations**

- **Advanced Text Processing:** TF-IDF vectorization, brand extraction, semantic clustering
- **Multi-Modal Features:** Combines text and image features for better predictions
- **Memory Optimization:** Chunked processing prevents system crashes
- **Robust Error Handling:** Graceful fallbacks for missing images/data
- **Flexible Architecture:** Multiple execution modes for different use cases

### **File Structure After Execution**

```
Vani_Amazon_ML_Challenge_25/
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── test_out.csv          # Generated predictions
│   └── test_out_correct.csv
├── models/
│   ├── trained_model.pkl     # LightGBM model
│   ├── tfidf_vectorizer.pkl  # Text vectorizer
│   ├── feature_columns.pkl   # Feature metadata
│   └── model_metadata.pkl    # Training info
├── images/                   # Downloaded images (LOCAL mode)
│   ├── train/
│   └── test/
└── src/
    ├── text_features.py
    ├── image_features.py
    ├── image_features_drive.py
    └── drive_utils_fast.py
```

### **Mode Selection Guide**

| Mode | Time | Memory Usage | Accuracy | Dataset Size | Use Case |
|------|------|-------------|----------|--------------|----------|
| **Local** | 2-4 hours | High (16GB+) | Highest | Full (50k) | Production/Best Results |
| **Light** | 30-60 min | Low (4-8GB) | Good | Subset (5k) | Quick Testing/Limited RAM |
| **Cloud** | 1-2 hours | Very High (32GB+) | High | Full (50k) | Cloud Computing |

### **Feature Comparison**

| Feature | Local | Light | Cloud |
|---------|-------|-------|-------|
| Text Processing | ✅ Full TF-IDF | ✅ Full TF-IDF | ✅ Full TF-IDF |
| Image Processing | ✅ Local Files | ✅ Drive API | ✅ Drive API |
| Deep Learning | ✅ ResNet50 | ❌ Traditional Only | ✅ ResNet50 |
| Semantic Clustering | ✅ Yes | ✅ Yes | ✅ Yes |
| Brand Extraction | ✅ Yes | ✅ Yes | ✅ Yes |
| Premium Detection | ✅ Yes | ✅ Yes | ✅ Yes |
| Multiprocessing | ✅ Yes | ✅ Limited | ✅ Yes |
| Google Drive | ❌ No | ✅ Yes | ✅ Yes |
| Memory Efficiency | ❌ High Usage | ✅ Optimized | ⚠️ Very High | 

### **Troubleshooting**

#### **Memory Issues**
- **"zsh: killed" error:** System OOM killer terminated process
- **Solution:** Use Light Mode (`train_model_drive_light.py`)
- **Alternative:** Reduce batch_size in image processing functions
- **Check RAM:** Ensure 8GB+ available for Light mode, 16GB+ for full modes

#### **Image Download Failures**
- **Local Mode:** Re-run `python data_setup.py` - handles retries automatically
- **Drive Mode:** Check internet connection and Google Drive API limits
- **Solution:** Script continues with zero-filled features for failed downloads

#### **Model Loading Errors**
- **Error:** "Trained model not found"
- **Solution:** Ensure training phase completed successfully before prediction
- **Check:** Verify `models/` folder contains `.pkl` files

#### **Performance Issues**
- **Slow execution:** Start with Light Mode, then upgrade if needed
- **GPU not detected:** TensorFlow will fallback to CPU automatically
- **Drive API slow:** Consider downloading images locally first

#### **Dependency Issues**
- **Missing packages:** `pip install -r requirements.txt` in virtual environment
- **TensorFlow errors:** `pip install tensorflow` for deep learning features
- **spaCy model:** `python -m spacy download en_core_web_sm`
- **OpenCV issues:** `pip install opencv-python`

#### **Google Drive Setup**
- **Authentication:** Update `config.py` with valid `credentials.json`
- **Folder access:** Ensure folder ID has public read permissions
- **API limits:** Google Drive API has daily quotas - use sparingly

#### **Common Error Messages**
```bash
# Memory exhaustion
zsh: killed → Use Light Mode

# Missing model
FileNotFoundError: trained_model.pkl → Run training first

# TensorFlow GPU
Could not load dynamic library → CPU fallback (normal)

# Drive API
HttpError 403 → Check credentials and permissions
```

---

## **Quick Start Recommendations**

1. **First Time Users:** Start with Light Mode
2. **Limited Memory (<8GB):** Use Light Mode only
3. **Best Accuracy:** Use Local Mode with full dataset
4. **Cloud Processing:** Use Cloud Mode with high-memory instances

## **Performance Expectations**

- **Light Mode SMAPE:** ~15-20% (good for quick testing)
- **Full Mode SMAPE:** ~10-15% (production quality)
- **Processing Time:** Light (30min) vs Full (2-4 hours)
- **Memory Usage:** Light (4-8GB) vs Full (16-32GB)