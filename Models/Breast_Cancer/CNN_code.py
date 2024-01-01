# Import necessary libraries
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

# Set a random seed for reproducibility
random.seed(100)

# Read DICOM data from a CSV file
dicom_data = pd.read_csv('/home/darshkaws/Documents/Multimodal_Project/Data/cbis-ddsm-breast-cancer-image/csv/dicom_info.csv')
image_dir = '/home/darshkaws/Documents/Multimodal_Project/Data/cbis-ddsm-breast-cancer-image/jpeg'

# Display the first few rows of the DICOM data
dicom_data.head()

# Display information about the DICOM data
dicom_data.info()

# Extract paths to cropped images from the DICOM data
cropped_images = dicom_data[dicom_data.SeriesDescription == 'cropped images'].image_path
cropped_images.head()

# Replace paths to cropped images with full paths
cropped_images = cropped_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
cropped_images.head()

# Display the first 10 cropped images
for file in cropped_images[0:10]:
    cropped_images_show = PIL.Image.open(file)
    gray_img = cropped_images_show.convert("L")
    plt.imshow(gray_img, cmap='gray')

# Extract paths to full mammogram images from the DICOM data
full_mammogram_images = dicom_data[dicom_data.SeriesDescription == 'full mammogram images'].image_path
full_mammogram_images.head()

# Replace paths to full mammogram images with full paths
full_mammogram_images = full_mammogram_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
full_mammogram_images.head()

# Display the first 10 full mammogram images
for file in full_mammogram_images[0:10]:
    full_mammogram_images_show = PIL.Image.open(file)
    gray_img = full_mammogram_images_show.convert("L")
    plt.imshow(gray_img, cmap='gray')

# Extract paths to ROI mask images from the DICOM data
ROI_mask_images = dicom_data[dicom_data.SeriesDescription == 'ROI mask images'].image_path
ROI_mask_images.head()

# Replace paths to ROI mask images with full paths
ROI_mask_images = ROI_mask_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
ROI_mask_images.head()

# Display the first 10 ROI mask images
for file in ROI_mask_images[0:10]:
    ROI_mask_images_show = PIL.Image.open(file)
    gray_img = ROI_mask_images_show.convert("L")
    plt.imshow(gray_img, cmap='gray')

# Read CSV files containing training data for calcification and mass cases
calc_case_train = pd.read_csv('/home/darshkaws/Documents/Multimodal_Project/Data/cbis-ddsm-breast-cancer-image/csv/calc_case_description_train_set.csv')
mass_case_train = pd.read_csv('/home/darshkaws/Documents/Multimodal_Project/Data/cbis-ddsm-breast-cancer-image/csv/mass_case_description_train_set.csv')

# Display the first 5 rows of calcification case data
calc_case_train.head(5)

# Display the first 5 rows of mass case data
mass_case_train.head(5)

# Create a copy of the DICOM data for cleaning
dicom_cleaned_data = dicom_data.copy()
dicom_cleaned_data.head()

# Drop unnecessary columns from the DICOM data
dicom_cleaned_data.drop(['PatientBirthDate', 'AccessionNumber', 'Columns', 'ContentDate', 'ContentTime', 'PatientSex', 'PatientBirthDate', 'ReferringPhysicianName', 'Rows', 'SOPClassUID', 'SOPInstanceUID', 'StudyDate', 'StudyID', 'StudyInstanceUID', 'StudyTime', 'InstanceNumber', 'SeriesInstanceUID', 'SeriesNumber'], axis=1, inplace=True)

# Display information about the cleaned DICOM data
dicom_cleaned_data.info()

# Check for missing values in the cleaned DICOM data
dicom_cleaned_data.isna().sum()

# Fill missing values in 'SeriesDescription' and 'Laterality' columns with backward fill
dicom_cleaned_data['SeriesDescription'].fillna(method='bfill', axis=0, inplace=True)
dicom_cleaned_data['Laterality'].fillna(method='bfill', axis=0, inplace=True)

# Check for missing values again
dicom_cleaned_data.isna().sum()

# Create a copy of calcification case data for cleaning
Data_cleaning_1 = calc_case_train.copy()

# Rename columns for consistency
Data_cleaning_1 = Data_cleaning_1.rename(columns={
    'calc type': 'calc_type',
    'calc distribution': 'calc_distribution',
    'image view': 'image_view',
    'left or right breast': 'left_or_right_breast',
    'breast density': 'breast_density',
    'abnormality type': 'abnormality_type'
})

# Define categorical columns
categorical_cols = ['pathology', 'calc_type', 'calc_distribution',
                    'abnormality_type', 'image_view', 'left_or_right_breast']

# Convert specified columns to categorical data type
Data_cleaning_1[categorical_cols] = Data_cleaning_1[categorical_cols].astype('category')

# Check for missing values in the cleaned calcification case data
Data_cleaning_1.isna().sum()

# Fill missing values in 'calc_type' and 'calc_distribution' columns with backward fill
Data_cleaning_1['calc_type'].fillna(method='bfill', axis=0, inplace=True)
Data_cleaning_1['calc_distribution'].fillna(method='bfill', axis=0, inplace=True)

# Check for missing values again
Data_cleaning_1.isna().sum()

# Create a copy of mass case data for cleaning
Data_cleaning_2 = mass_case_train.copy()

# Rename columns for consistency
Data_cleaning_2 = Data_cleaning_2.rename(columns={
    'mass shape': 'mass_shape',
    'left or right breast': 'left_or_right_breast',
    'mass margins': 'mass_margins',
    'image view': 'image_view',
    'abnormality type': 'abnormality_type'
})

# Define categorical columns
categorical_cols = ['left_or_right_breast', 'image_view', 'mass_margins',
                    'mass_shape', 'abnormality_type', 'pathology']

# Convert specified columns to categorical data type
Data_cleaning_2[categorical_cols] = Data_cleaning_2[categorical_cols].astype('category')

# Check for missing values in the cleaned mass case data
Data_cleaning_2.isna().sum()

# Fill missing values in 'mass_shape' and 'mass_margins' columns using backward fill (bfill)
Data_cleaning_2['mass_shape'].fillna(method='bfill', axis=0, inplace=True)
Data_cleaning_2['mass_margins'].fillna(method='bfill', axis=0, inplace=True)

# Check and display the count of missing values in the DataFrame
Data_cleaning_2.isna().sum()

# List all image file paths recursively within the specified directory
breast_imgs = glob.glob('/home/darshkaws/Documents/Multimodal_Project/Data/breast-histopathology-images/IDC_regular_ps50_idx5/**/*.png', recursive=True)

# Print the first 5 image file paths
for imgname in breast_imgs[:5]:
    print(imgname)

# Initialize empty lists to store non-cancerous and cancerous image paths
non_cancer_imgs = []
cancer_imgs = []

# Categorize image paths based on the last character of the file name (0 for non-cancer, 1 for cancer)
for img in breast_imgs:
    if img[-5] == '0':
        non_cancer_imgs.append(img)
    elif img[-5] == '1':
        cancer_imgs.append(img)

# Calculate the number of non-cancer and cancer images
non_cancer_num = len(non_cancer_imgs)  # No cancer
cancer_num = len(cancer_imgs)  # Cancer

# Calculate the total number of images
total_img_num = non_cancer_num + cancer_num

# Display the counts of non-cancer, cancer, and total images
print('Number of Images of no cancer: {}'.format(non_cancer_num))  # Images of Non-cancer
print('Number of Images of cancer: {}'.format(cancer_num))   # Images of cancer
print('Total Number of Images: {}'.format(total_img_num))

# Create a DataFrame for insights on the state of cancer patients
data_insight_1 = pd.DataFrame({'State of Cancer': ['0', '1'], 'Numbers of Patients': [397476, 157572]})

# Create a bar plot to visualize the number of patients with and without cancer
bar = px.bar(data_frame=data_insight_1, x='State of Cancer', y='Numbers of Patients', color='State of Cancer')
bar.update_layout(title_text='Number of patients with (1) and without (0) cancer', title_x=0.5)
bar.show()

# Get unique series descriptions from dicom_cleaned_data
series_descriptions = dicom_cleaned_data['SeriesDescription'].unique()

# Define colors for the bar chart
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'grey']

# Create a bar chart for different image types
fig = go.Figure()
for i, desc in enumerate(series_descriptions):
    # Filter data based on series description
    filtered_data = dicom_cleaned_data[dicom_cleaned_data['SeriesDescription'] == desc]
    # Add a bar trace for each series description
    fig.add_trace(go.Bar(
        x=[desc],
        y=[len(filtered_data)],
        name=desc,
        marker_color=colors[i % len(colors)]
    ))

# Update layout and display the bar chart
fig.update_layout(
    title_text='Image Types',
    title_x=0.5,
    xaxis_title='Series Description',
    yaxis_title='Count',
    barmode='group')
fig.show()

# Create a DataFrame to analyze the count of different breast parts examined
f = pd.DataFrame(dicom_cleaned_data['BodyPartExamined'].value_counts())
f = f.reset_index()
f = f.rename(columns={'index': 'Breast_Part_Examined', 'BodyPartExamined': 'Count'})

# Create a bar plot to visualize the count of different breast parts examined
bar = px.bar(data_frame=f, x="Breast_Part_Examined", y="Count", color="Breast_Part_Examined")
bar.update_layout(title_text='Breast Parts Examined', title_x=0.5, yaxis=dict(type='log'))
bar.show()

# Create a DataFrame to analyze the count of abnormality types
data_insight_2 = pd.DataFrame({'abnormality': [Data_cleaning_1.abnormality_type[0], Data_cleaning_2.abnormality_type[0]],
                               'counts_of_abnormalities': [len(Data_cleaning_1), len(Data_cleaning_2)]})

# Create a bar plot to visualize the count of abnormality types
bar_2 = px.bar(data_frame=data_insight_2, x='abnormality', y='counts_of_abnormalities', color='abnormality')
bar_2.update_layout(title_text='Abnormality Type', title_x=0.5)
bar_2.show()

# Calculate the counts of left and right breasts in Data_cleaning_1
x = Data_cleaning_1.left_or_right_breast.value_counts().RIGHT
y = Data_cleaning_1.left_or_right_breast.value_counts().LEFT

# Create a DataFrame to analyze the count of left and right breasts
data_insight_3 = pd.DataFrame({'left_or_right_breast': ['RIGHT', 'LEFT'], 'Counts': [x, y]})

# Create a bar plot to visualize the count of left and right breasts
insight_3 = px.bar(data_insight_3, y='Counts', x='left_or_right_breast', color='left_or_right_breast')
insight_3.update_layout(title_text='Calcification position (left or right breast)', title_x=0.5)
insight_3.show()

# Create a DataFrame to analyze the count of calcification types in Data_cleaning_1
z = pd.DataFrame(Data_cleaning_1['calc_type'].value_counts())
z = z.reset_index()
z = z.rename(columns={'calc_type': 'calc_type_counts'})

# Display the DataFrame containing counts of calcification types
z

# Set up a matplotlib figure for image visualization
plt.figure(figsize=(15, 15))

# Randomly select 18 non-cancerous and 18 cancerous image indices for display
some_non_cancerous = np.random.randint(0, len(non_cancer_imgs), 18)
some_cancerous = np.random.randint(0, len(cancer_imgs), 18)

s = 0
# Display 18 random non-cancerous images
for num in some_non_cancerous:
    img = image.load_img((non_cancer_imgs[num]), target_size=(100, 100))
    img = image.img_to_array(img)
    plt.subplot(6, 6, 2 * s + 1)
    plt.axis('off')
    plt.title('no cancer')
    plt.imshow(img.astype('uint8'))
    s += 1

s = 1
# Display 18 random cancerous images
for num in some_cancerous:
    
        img = image.load_img((cancer_imgs[num]), target_size=(100, 100))
        img = image.img_to_array(img)
        plt.subplot(6, 6, 2*s)
        plt.axis('off')        
        plt.title('cancer')
        plt.imshow(img.astype('uint8'))
        s += 1

# Randomly sample images from two lists, 'non_cancer_imgs' and 'cancer_imgs'
some_non_img = random.sample(non_cancer_imgs, len(non_cancer_imgs))
some_can_img = random.sample(cancer_imgs, len(cancer_imgs))

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
    # Append the image data (feature) to the 'X' list
    X.append(feature)
    # Append the label to the 'y' list
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
y_train = to_categorical(y_train, 2)  # Assuming there are 2 classes (non-cancer and cancer)
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
    patience=20,                  # Number of epochs with no improvement after which training will be stopped
    min_delta=1e-7,              # Minimum change in the monitored quantity to be considered an improvement
    restore_best_weights=True,   # Restore model weights from the epoch with the best value of monitored quantity
)

# Define a ReduceLROnPlateau callback
plateau = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',   # Monitor the validation loss
    factor=0.2,           # Factor by which the learning rate will be reduced (new_lr = lr * factor)
    patience=5,           # Number of epochs with no improvement after which learning rate will be reduced
    min_delta=1e-7,       # Minimum change in the monitored quantity to trigger a learning rate reduction
    cooldown=0,           # Number of epochs to wait before resuming normal operation after learning rate reduction
    verbose=1             # Verbosity mode (1: update messages, 0: no messages)
)

# Set a random seed for reproducibility
tf.random.set_seed(42)

# Create a Sequential model
model = keras.Sequential([
    # Convolutional layer with 32 filters, a 3x3 kernel, 'same' padding, and ReLU activation
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(50, 50, 3)),
    keras.layers.BatchNormalization(),
    # MaxPooling layer with a 2x2 pool size and default stride (2)
    keras.layers.MaxPooling2D(strides=2),
    
    # Convolutional layer with 64 filters, a 3x3 kernel, 'same' padding, and ReLU activation
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    # MaxPooling layer with a 3x3 pool size and stride of 2
    keras.layers.MaxPooling2D((3, 3), strides=2),
    
    # Convolutional layer with 128 filters, a 3x3 kernel, 'same' padding, and ReLU activation
    keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    # MaxPooling layer with a 3x3 pool size and stride of 2
    keras.layers.MaxPooling2D((3, 3), strides=2),
    
    # Convolutional layer with 128 filters, a 3x3 kernel, 'same' padding, and ReLU activation
    keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    # MaxPooling layer with a 3x3 pool size and stride of 2
    keras.layers.MaxPooling2D((3, 3), strides=2),
    
    # Flatten the output to prepare for fully connected layers
    keras.layers.Flatten(),
    
    # Fully connected layer with 128 units and ReLU activation
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    # Output layer with 2 units (binary classification) and softmax activation
    keras.layers.Dense(2, activation='softmax')
])

# Display a summary of the model architecture
model.summary()

# Compile the model with Adam optimizer, binary cross-entropy loss, and accuracy metric
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train[:55000],
                    y_train[:55000],
                    validation_data = (X_test[:14000], y_test[:14000]),
                    epochs = 28,
                    batch_size = 75,
                    callbacks=[early_stopping, plateau])


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

# Define a mapping of calcification types
calcification_types = {
    0: 'No Calcification',
    1: 'PLEOMORPHIC',
    2: 'AMORPHOUS',
    3: 'PUNCTATE',
    4: 'LUCENT_CENTER',
    5: 'VASCULAR',
    6: 'FINE_LINEAR_BRANCHING',
    7: 'COARSE',
    8: 'ROUND_AND_REGULAR-LUCENT_CENTER',
    9: 'PLEOMORPHIC-FINE_LINEAR_BRANCHING',
    10: 'ROUND_AND_REGULAR-LUCENT_CENTER-PUNCTATE',
    11: 'ROUND_AND_REGULAR-EGGSHELL',
    12: 'PUNCTATE-PLEOMORPHIC',
    13: 'DYSTROPHIC',
    14: 'LUCENT_CENTERED',
    15: 'ROUND_AND_REGULAR-LUCENT_CENTER-DYSTROPHIC',
    16: 'ROUND_AND_REGULAR',
    17: 'ROUND_AND_REGULAR-LUCENT_CENTERED',
    18: 'AMORPHOUS-PLEOMORPHIC',
    19: 'LARGE_RODLIKE-ROUND_AND_REGULAR',
    20: 'PUNCTATE-AMORPHOUS',
    21: 'COARSE-ROUND_AND_REGULAR-LUCENT_CENTER',
    22: 'VASCULAR-COARSE-LUCENT_CENTERED',
    23: 'LUCENT_CENTER-PUNCTATE',
    24: 'ROUND_AND_REGULAR-PLEOMORPHIC',
    25: 'EGGSHELL',
    26: 'PUNCTATE-FINE_LINEAR_BRANCHING',
    27: 'VASCULAR-COARSE',
    28: 'ROUND_AND_REGULAR-PUNCTATE',
    29: 'SKIN-PUNCTATE-ROUND_AND_REGULAR',
    30: 'SKIN-PUNCTATE',
    31: 'COARSE-ROUND_AND_REGULAR-LUCENT_CENTERED',
    32: 'PUNCTATE-ROUND_AND_REGULAR',
    33: 'LARGE_RODLIKE',
    34: 'AMORPHOUS-ROUND_AND_REGULAR',
    35: 'PUNCTATE-LUCENT_CENTER',
    36: 'SKIN',
    37: 'VASCULAR-COARSE-LUCENT_CENTER-ROUND_AND_REGULA',
    38: 'COARSE-PLEOMORPHIC',
    39: 'ROUND_AND_REGULAR-PUNCTATE-AMORPHOUS',
    40: 'COARSE-LUCENT_CENTER',
    41: 'MILK_OF_CALCIUM',
    42: 'COARSE-ROUND_AND_REGULAR',
    43: 'SKIN-COARSE-ROUND_AND_REGULAR',
    44: 'ROUND_AND_REGULAR-AMORPHOUS',
    45: 'PLEOMORPHIC-PLEOMORPHIC'
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

# Get the calcification type based on the predicted class index (modify as needed)
calcification_type = calcification_types[predicted_class_index]

# Print the prediction result with calcification type
print('Predicted Diagnosis:', predicted_label)
print('Calcification Type:', calcification_type)
print('True Diagnosis:', true_label)

model.save('/home/darshkaws/Documents/Multimodal_Project/Code/Models/CNN_model_4.keras')

