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



model = keras.models.load_model('/home/darshkaws/Documents/Multimodal_Project/Code/Models/CNN_model_fo.keras')

breast_imgs = glob.glob('/home/darshkaws/Documents/Multimodal_Project/Data/breast-histopathology-images/**/*.png', recursive = True)

for imgname in breast_imgs[:5]:
    print(imgname)


non_cancer_imgs = []
cancer_imgs = []

for img in breast_imgs:
    if img[-5] == '0' :
        non_cancer_imgs.append(img)
    
    elif img[-5] == '1' :
        cancer_imgs.append(img)

non_cancer_num = len(non_cancer_imgs)  # No cancer
cancer_num = len(cancer_imgs)   # Cancer 
        
total_img_num = non_cancer_num + cancer_num
        
print('Number of Images of no cancer: {}' .format(non_cancer_num))   # images of Non cancer
print('Number of Images of cancer : {}' .format(cancer_num))   # images of cancer 
print('Total Number of Images : {}' .format(total_img_num))


# Randomly sample images from two lists, 'non_cancer_imgs' and 'cancer_imgs'
some_non_img = random.sample(non_cancer_imgs, len(non_cancer_imgs))
some_can_img = random.sample(cancer_imgs, len(cancer_imgs))

non_img_arr = []
can_img_arr = []

# Define a mapping of class indices to human-readable labels

class_labels = {
    0: 'Non-Cancerous',
    1: 'Cancerous',
}
for img_path in some_non_img:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is not None and img.size > 0:
        resized_img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR)
        non_img_arr.append([resized_img, 0])
    else:
        print(f"Warning: Unable to read image at {img_path}")

for img_path in some_can_img:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is not None and img.size > 0:
        resized_img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR)
        can_img_arr.append([resized_img, 1])
    else:
        print(f"Warning: Unable to read image at {img_path}")

# Convert lists to numpy arrays
non_img_arr = np.array(non_img_arr, dtype=object)
can_img_arr = np.array(can_img_arr, dtype=object)


breast_img_arr = np.concatenate((non_img_arr, can_img_arr))
    
X = []  # List for image data
y = []  # List for labels

C = []
CL = []

NC = []
NCL = []

# Shuffle the elements in the 'breast_img_arr' array randomly
random.shuffle(breast_img_arr)

for feature, label in breast_img_arr:
    # Append the image data (feature) to the 'X' list
    X.append(feature)
    y.append(label)

for feature, label in can_img_arr:
    # Append the image data (feature) to the 'X' list
    C.append(feature)
    CL.append(label)
    
for feature, label in non_img_arr:
    # Append the image data (feature) to the 'X' list
    NC.append(feature)
    # Append the label to the 'y' list
    NCL.append(label)

# Convert the lists 'X' and 'y' into NumPy arrays
X = np.array(X)
y = np.array(y)

C = np.array(C)
CL = np.array(CL)

NC = np.array(NC)
NCL = np.array(NCL)

print('X shape: {}'.format(X.shape))

count_c = 0
count_n = 0

count_nc = 0
count_cn = 0

for index in range (0, 1000): 
    # Extract a single image from X_test based on the specified index
    input = C[index:index+1]

    # Make a prediction using the CNN model and get the class with the highest probability
    predicted_class_index = model.predict(input)[0].argmax()
    predicted_label = class_labels[predicted_class_index]

    print('Predicted Diagnosis:', predicted_label)
    if predicted_label == "Cancerous":
        count_c += 1
    else:
        count_n += 1
        
for index in range (0, 1000): 
    # Extract a single image from X_test based on the specified index
    input = NC[index:index+1]

    # Make a prediction using the CNN model and get the class with the highest probability
    predicted_class_index = model.predict(input)[0].argmax()
    predicted_label = class_labels[predicted_class_index]

    print('Predicted Diagnosis:', predicted_label)
    if predicted_label == "Non-Cancerous":
        count_nc += 1
    else:
        count_cn += 1
    
Sensitivity = count_c/1000
print(f"Cases detected successfully: {count_c}")
print(f"Cases detected unsuccessfully: {count_n}")
print(f"Sensitivity: {count_c/1000}")
        
Specificity = count_nc/1000
print(f"Cases detected successfully: {count_nc}")
print(f"Cases detected unsuccessfully: {count_cn}")
print(f"Specificity: {count_nc/1000}")

plt.subplot(1, 2, 1)
plt.bar(['Sensitivity', 'Specificity'], [Sensitivity, Specificity], width=0.8, color = "maroon")

plt.title("Detection Rates")
plt.xlabel("Accuracy Measures")
plt.ylabel("In Percentage (%)")
plt.show()