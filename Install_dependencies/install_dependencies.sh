#!/bin/bash

# Update pip
python3 -m pip install --upgrade pip

# Install required packages
pip3 install pandas numpy matplotlib plotly seaborn opencv-python tensorflow keras scikit-learn Pillow

# Check if the installation was successful
if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully."
else
    echo "Failed to install dependencies."
fi