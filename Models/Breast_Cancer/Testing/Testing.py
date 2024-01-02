# Testing the model
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objs as go

import cv2
from matplotlib.image import imread

import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import glob
import PIL
from PIL import Image
import random
import os
import pandas as pd
import random
import glob

plt.style.use('dark_background')

model = keras.models.load_model('/home/darshkaws/Documents/Multimodal_Project/Code/Models/CNN_model_7.keras')

images = glob.glob('/home/darshkaws/Documents/Multimodal_Project/Data/breast-histopathology-images/**/*.png', recursive=True)

n = min(5, len(images))  
selected_images = random.sample(images, n)

def load_and_preprocess_image(image_path, target_size=(50, 50)):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            return resized_img
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
    return None

# Process and store the selected images
preprocessed_images = [load_and_preprocess_image(img_path) for img_path in selected_images]

# Filter out None values
preprocessed_images = [img for img in preprocessed_images if img is not None]

# Only select corresponding images from selected_images list
valid_image_paths = [img_path for i, img_path in enumerate(selected_images) if preprocessed_images[i] is not None]

# Convert to numpy array
img_arr = np.array(preprocessed_images)

# Assuming the model expects a batch of images
predictions = model.predict(img_arr)

# Display each image with its prediction and true label
for i, img in enumerate(img_arr):
    plt.subplot(1, len(img_arr), i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert to RGB for display

    cancer_probability = predictions[i][0]
    print(cancer_probability)
    predicted_class = "MALIGNANT" if cancer_probability <= 0.5 else "BENIGN"
    
    if valid_image_paths[i][-5] == '0':
        pathology_results = "BENIGN"
    elif valid_image_paths[i][-5] == '1':
        pathology_results = "MALIGNANT"
    print(valid_image_paths[i])
    plt.title(f"True Pathology: {pathology_results} \n Predicted Pathology: {predicted_class}\nProbability of Cancer: {1-cancer_probability:.4f}", fontsize = 10)

plt.show()