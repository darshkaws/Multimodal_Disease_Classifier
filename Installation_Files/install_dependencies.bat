@echo off

:: Install Python dependencies.
pip install pandas numpy matplotlib plotly seaborn opencv-python tensorflow keras scikit-learn Pillow

:: Check if the installation was successful.
if %errorlevel%==0 (
    echo Dependencies installed successfully.
) else (
    echo Failed to install dependencies.
)

:: Pause to keep the command window open.
pause