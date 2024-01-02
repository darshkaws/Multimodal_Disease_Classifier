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
import random

random.seed(100)

breast_imgs = glob.glob('/home/darshkaws/Documents/Multimodal_Project/Data/breast-histopathology-images/IDC_regular_ps50_idx5/**/*.png', recursive = True)

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

plt.figure(figsize = (15, 15))

some_non_cancerous = np.random.randint(0, len(non_cancer_imgs), 18)
some_cancerous = np.random.randint(0, len(cancer_imgs), 18)

s = 0
for num in some_non_cancerous:
    
        img = image.load_img((non_cancer_imgs[num]), target_size=(100, 100))
        img = image.img_to_array(img)
        
        plt.subplot(6, 6, 2*s+1)
        plt.axis('off')
        plt.title('no cancer')
        plt.imshow(img.astype('uint8'))
        s += 1
        
s = 1

for num in some_cancerous:
    
        img = image.load_img((cancer_imgs[num]), target_size=(100, 100))
        img = image.img_to_array(img)
        plt.subplot(6, 6, 2*s)
        plt.axis('off')        
        plt.title('cancer')
        plt.imshow(img.astype('uint8'))
        s += 1

# Randomly sample images from two lists, 'non_cancer_imgs' and 'cancer_imgs'
some_non_img = random.sample(non_cancer_imgs, 70000)
some_can_img = random.sample(cancer_imgs, 65000)

non_img_arr = []
can_img_arr = []

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

# Shuffle the elements in the 'breast_img_arr' array randomly
random.shuffle(breast_img_arr)

# Loop through each element (feature, label) in the shuffled 'breast_img_arr'
for feature, label in breast_img_arr:
    X.append(feature)
    y.append(label)

# Convert the lists 'X' and 'y' into NumPy arrays
X = np.array(X)
y = np.array(y)

print('X shape: {}'.format(X.shape))



# Split the dataset into training and testing sets, with a test size of 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Define a rate (percentage) for subsampling the training data
rate = 0.5

# Calculate the number of samples to keep in the training data based on the rate
num = int(X.shape[0] * rate)

# Convert the categorical labels in 'y_train' and 'y_test' to one-hot encoded format
y_train = to_categorical(y_train, 2)  
y_test = to_categorical(y_test, 2)


print('X_train shape : {}' .format(X_train.shape))
print('X_test shape : {}' .format(X_test.shape))
print('y_train shape : {}' .format(y_train.shape))
print('y_test shape : {}' .format(y_test.shape))

from keras.preprocessing.image import ImageDataGenerator
# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create data generators for training and testing
train_datagen = datagen.flow(X_train, y_train, batch_size=32)
test_datagen = datagen.flow(X_test, y_test, batch_size=32, shuffle=False)



# Define an EarlyStopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',          # Monitor the validation loss
    patience=15,                  # Number of epochs with no improvement after which training will be stopped
)


tf.random.set_seed(42)

# Create a Sequential model
model = keras.Sequential([

keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(50, 50, 3)),
keras.layers.BatchNormalization(),
keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
keras.layers.MaxPooling2D((2, 2)),
keras.layers.BatchNormalization(),
keras.layers.Dropout(0.3),
keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
keras.layers.BatchNormalization(),
keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
keras.layers.BatchNormalization(),
keras.layers.MaxPooling2D((2, 2)),
keras.layers.Dropout(0.3),
keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
keras.layers.Flatten(),
keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
keras.layers.BatchNormalization(),
keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
keras.layers.BatchNormalization(),
keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
keras.layers.Dropout(0.3),
keras.layers.Dense(24, activation='relu', kernel_initializer='he_uniform'),
keras.layers.Dense(2, activation='softmax')

])

# Display a summary of the model architecture
model.summary()

# Compile the model with Adam optimizer, binary cross-entropy loss, and accuracy metric
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train[:65000],
                    y_train[:65000],
                    validation_data = (X_test[:17000], y_test[:17000]),
                    epochs = 40,
                    batch_size = 50,
                    callbacks=[early_stopping])


model.evaluate(X_test,y_test)

Y_pred = model.predict(X_test[:10000])
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(y_test[:10000],axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="BuPu",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

prediction = model.predict(X_test)
prediction

# Define a mapping of class indices to human-readable labels
class_labels = {
    0: 'Non-Cancerous',
    1: 'Cancerous',
}

# Define a function for plotting an image from an array
def img_plot(arr, index=0):
    # Set the title for the plot
    plt.title('Test Image')
    
    # Display the image at the specified index in the array
    plt.imshow(arr[index])

# Set the index value to 90
index = 90

# Plot an image from the X_test array using the img_plot function
img_plot(X_test, index)

# Extract a single image from X_test based on the specified index
input = X_test[index:index+1]

# Make a prediction using the CNN model and get the class with the highest probability
predicted_class_index = model.predict(input)[0].argmax()

# Get the true label from the y_test array
true_class_index = y_test[index].argmax()

# Get the predicted and true labels
predicted_label = class_labels[predicted_class_index]
true_label = class_labels[true_class_index]

print('Predicted Diagnosis:', predicted_label)
print('True Diagnosis:', true_label)

model.save('/home/darshkaws/Documents/Multimodal_Project/Code/Models/CNN_model_7.keras')

