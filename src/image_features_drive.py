import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageStat
from sklearn.cluster import KMeans
from src.drive_utils import get_image_from_drive, init_drive_loader

try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50, EfficientNetB0
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import img_to_array
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

IMAGE_SIZE = (224, 224)

def extract_color_features_drive(image_link):
    """Extract color features from Drive image"""
    try:
        img = get_image_from_drive(image_link)
        if img is None:
            raise ValueError("Could not load image from Drive")
            
        img = img.convert('RGB')
        stat = ImageStat.Stat(img)
        
        img_array = np.array(img.resize((50, 50)))
        pixels = img_array.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_
        
        return {
            'brightness': np.mean(stat.mean),
            'contrast': np.std(stat.mean),
            'color_variance': np.var(pixels, axis=0).mean(),
            'dominant_color_1_r': dominant_colors[0][0],
            'dominant_color_1_g': dominant_colors[0][1], 
            'dominant_color_1_b': dominant_colors[0][2],
            'dominant_color_2_r': dominant_colors[1][0],
            'dominant_color_2_g': dominant_colors[1][1],
            'dominant_color_2_b': dominant_colors[1][2],
            'color_richness': len(np.unique(pixels.reshape(-1, 3), axis=0)) / len(pixels)
        }
    except:
        return {f'brightness': 0, 'contrast': 0, 'color_variance': 0,
                'dominant_color_1_r': 0, 'dominant_color_1_g': 0, 'dominant_color_1_b': 0,
                'dominant_color_2_r': 0, 'dominant_color_2_g': 0, 'dominant_color_2_b': 0,
                'color_richness': 0}

def extract_texture_features_drive(image_link):
    """Extract texture features from Drive image"""
    try:
        img = get_image_from_drive(image_link)
        if img is None:
            raise ValueError("Could not load image from Drive")
            
        img_array = np.array(img.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        return {
            'texture_variance': np.var(gray),
            'edge_density': np.mean(cv2.Canny(gray, 50, 150)) / 255.0,
            'gradient_magnitude': np.mean(np.gradient(gray.astype(float))),
        }
    except:
        return {'texture_variance': 0, 'edge_density': 0, 'gradient_magnitude': 0}

def extract_composition_features_drive(image_link):
    """Extract composition features from Drive image"""
    try:
        img = get_image_from_drive(image_link)
        if img is None:
            raise ValueError("Could not load image from Drive")
            
        img_array = np.array(img.convert('RGB'))
        height, width = img_array.shape[:2]
        
        center_h, center_w = height//4, width//4
        center_region = img_array[center_h:3*center_h, center_w:3*center_w]
        
        return {
            'aspect_ratio': width / height,
            'center_brightness': np.mean(center_region),
            'center_vs_edge_contrast': np.mean(center_region) - np.mean(img_array),
            'image_complexity': np.std(img_array),
        }
    except:
        return {'aspect_ratio': 1.0, 'center_brightness': 0, 'center_vs_edge_contrast': 0, 'image_complexity': 0}

def load_and_preprocess_image_drive(image_link):
    """Load and preprocess image from Drive"""
    try:
        img = get_image_from_drive(image_link)
        if img is None:
            return None
            
        img = img.convert('RGB').resize(IMAGE_SIZE)
        return img_to_array(img)
    except:
        return None

def extract_deep_features_drive(df, model_name='resnet50', batch_size=32):
    """Extract deep features from Drive images"""
    if not TF_AVAILABLE:
        return np.zeros((len(df), 2048))
    
    if model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        preprocess_func = resnet_preprocess
        feature_size = 2048
    else:
        base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
        preprocess_func = efficientnet_preprocess
        feature_size = 1280
    
    model = Model(inputs=base_model.input, outputs=base_model.output)
    
    image_list = []
    for link in tqdm(df['image_link'], desc="Loading Drive Images"):
        if pd.notna(link):
            img_array = load_and_preprocess_image_drive(link)
            image_list.append(img_array if img_array is not None else np.zeros(IMAGE_SIZE + (3,)))
        else:
            image_list.append(np.zeros(IMAGE_SIZE + (3,)))
    
    if not image_list:
        return np.zeros((len(df), feature_size))
    
    image_batch = np.array(image_list)
    image_batch = preprocess_func(image_batch)
    
    features = model.predict(image_batch, batch_size=batch_size, verbose=1)
    return features

def extract_comprehensive_image_features_drive(df: pd.DataFrame, use_deep_features=True, model_name='resnet50', 
                                             credentials_path='credentials.json', folder_id='1ZXP3slTxtjvVaqTFrblfR8eR5lf07nNK'):
    """Extract comprehensive image features from Google Drive"""
    
    # Initialize Drive loader
    init_drive_loader(credentials_path, folder_id)
    
    print("🚀 Starting Drive Image Feature Extraction...")
    
    # Extract traditional features
    color_features_list = []
    texture_features_list = []
    composition_features_list = []
    
    for link in tqdm(df['image_link'], desc="Extracting Features"):
        if pd.notna(link):
            color_features_list.append(extract_color_features_drive(link))
            texture_features_list.append(extract_texture_features_drive(link))
            composition_features_list.append(extract_composition_features_drive(link))
        else:
            color_features_list.append(extract_color_features_drive(None))
            texture_features_list.append(extract_texture_features_drive(None))
            composition_features_list.append(extract_composition_features_drive(None))
    
    # Combine traditional features
    traditional_features = []
    for i in range(len(df)):
        combined = {**color_features_list[i], **texture_features_list[i], **composition_features_list[i]}
        traditional_features.append(list(combined.values()))
    
    traditional_features = np.array(traditional_features)
    
    # Extract deep features if requested
    if use_deep_features:
        deep_features = extract_deep_features_drive(df, model_name=model_name)
        final_features = np.hstack([traditional_features, deep_features])
    else:
        final_features = traditional_features
    
    print(f"✅ Drive image feature extraction complete! Shape: {final_features.shape}")
    
    feature_names = (list(color_features_list[0].keys()) + 
                    list(texture_features_list[0].keys()) +
                    list(composition_features_list[0].keys()))
    
    if use_deep_features:
        feature_names.extend([f'{model_name}_feature_{i}' for i in range(deep_features.shape[1])])
    
    return final_features, feature_names