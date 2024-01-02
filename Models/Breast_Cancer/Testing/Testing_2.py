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

import scipy.stats as stats
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

breast_imgs = glob.glob('/home/darshkaws/Documents/Multimodal_Project/Data/breast-histopathology-images/**/*.png', recursive=True)
class_labels = {0: 'Non-Cancerous', 1: 'Cancerous'}

def run_test(model, breast_imgs, num_samples=300, num_iterations=30):
    sensitivity_list = []
    specificity_list = []

    for _ in range(num_iterations):
        random.shuffle(breast_imgs)
        selected_imgs = breast_imgs[:num_samples]
        tp, tn, fp, fn = 0, 0, 0, 0

        for img_path in selected_imgs:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR)
            img = np.expand_dims(img, axis=0)
            predicted_class_index = model.predict(img)[0].argmax()
            true_class = int(img_path[-5])  # Assuming label is in filename

            if predicted_class_index == true_class == 1:
                tp += 1
            elif predicted_class_index == true_class == 0:
                tn += 1
            elif predicted_class_index == 1 and true_class == 0:
                fp += 1
            elif predicted_class_index == 0 and true_class == 1:
                fn += 1

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    return sensitivity_list, specificity_list

# Run the test multiple times
sensitivity_results, specificity_results = run_test(model, breast_imgs, num_samples=300, num_iterations=30)

# Calculate the mean and standard deviation
mean_sensitivity = np.mean(sensitivity_results)
std_dev_sensitivity = np.std(sensitivity_results)
mean_specificity = np.mean(specificity_results)
std_dev_specificity = np.std(specificity_results)

print("Mean Sensitivity:", mean_sensitivity, "Standard Deviation:", std_dev_sensitivity)
print("Mean Specificity:", mean_specificity, "Standard Deviation:", std_dev_specificity)

# Plotting
plt.figure(figsize=(12, 6))
sensitivity_range = np.linspace(0, 1, 100)
specificity_range = np.linspace(0, 1, 100)

plt.plot(sensitivity_range, stats.norm.pdf(sensitivity_range, mean_sensitivity, std_dev_sensitivity), label='Sensitivity Distribution', color='blue')
plt.plot(specificity_range, stats.norm.pdf(specificity_range, mean_specificity, std_dev_specificity), label='Specificity Distribution', color='red')
plt.title('Normal Distributions of Sensitivity and Specificity')
plt.legend()
plt.show()