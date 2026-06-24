# Week 2: Data Preprocessing and Feature Engineering

## Project Overview

This project focuses on data preprocessing and feature engineering using an automobile dataset. The objective is to clean raw data, handle potential data quality issues, create meaningful features, and prepare the dataset for machine learning applications.

## Dataset

The dataset contains automobile information such as:

* Car Name
* Fuel Type
* Engine Type
* Horsepower
* Engine Size
* City MPG
* Highway MPG
* Price
* Vehicle Dimensions

Dataset Size:

* Rows: 211
* Columns: 25

## Data Preprocessing Steps

### 1. Missing Value Analysis

* Checked the dataset for missing values.
* Visualized missing values using a heatmap.

### 2. Duplicate Removal

* Identified and removed duplicate records.

### 3. Outlier Detection and Removal

* Used the Interquartile Range (IQR) method to detect outliers in the Price column.
* Removed extreme values to improve data quality.

### 4. Encoding Categorical Variables

* Applied One-Hot Encoding to convert categorical features into numerical format.

### 5. Feature Scaling

* Standardized numerical features using StandardScaler.

## Feature Engineering

The following new features were created:

### Power Engine Ratio

Horsepower divided by Engine Size.

Purpose:

* Measures engine efficiency and performance.

### Fuel Economy Score

Average of City MPG and Highway MPG.

Purpose:

* Represents overall fuel efficiency.

### Price Per Horsepower

Price divided by Horsepower.

Purpose:

* Evaluates vehicle value relative to performance.

## Visualizations Generated

* Missing Values Heatmap
* Price Boxplot Before Outlier Removal
* Price Boxplot After Outlier Removal
* Fuel Economy Score Distribution
* Correlation Heatmap

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Visual Studio Code

## Project Files

* preprocessing.py
* automobile.csv
* processed_automobile.csv
* Week2_Data_Preprocessing_Report.docx
* outputs_missing_values.png
* boxplot_before.png
* boxplot_after.png
* feature_distribution.png
* correlation_heatmap.png

## Results

The preprocessing pipeline successfully:

* Improved data quality
* Removed duplicates and outliers
* Created meaningful engineered features
* Generated visual insights for analysis
* Prepared the dataset for machine learning applications

## Author

Week 2 Automobile Data Preprocessing and Feature Engineering Project
