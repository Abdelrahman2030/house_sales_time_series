# House Sales Time Series Prediction

## Overview
This project involves predicting house sales prices using a time series dataset enriched with additional features derived from the data. The goal was to build a machine learning model that effectively predicts sales prices and evaluates its performance based on RMSE.

---

## Dataset Overview
The dataset contains information about house sales, including features such as:
- `price`: The sale price of the house (target variable).
- `bedrooms`: Number of bedrooms.
- `post_code`: Postal code of the property.
- `property_type`: Type of property (e.g., detached, semi-detached).
- `date`: The date the house was sold.

---

## Preprocessing Steps
### 1. **Column Renaming**
- Renamed columns to make them more readable and consistent.

### 2. **Handling Missing Values**
- Missing values in the `bedrooms` column were filled with the median number of bedrooms.

### 3. **Outlier Identification**
- Added a new column to identify outliers in the `price` column using statistical techniques (e.g., IQR).

### 4. **Feature Engineering**
- Extracted time-based features from the `date` column:
  - `day`: Day of the sale.
  - `month`: Month of the sale.
  - `year`: Year of the sale.

### 5. **Data Encoding**
- **Label Encoding**:
  - Encoded the `bedrooms` column into numerical categories.
- **One-Hot Encoding**:
  - Applied to `post_code` and `property_type` columns.

### 6. **Data Transformation**
- Applied scaling to normalize the data.
- Performed logarithmic transformation on the `price` column to reduce skewness.

---

## Modeling
### Model Used
- **XGBoost Regressor**: A powerful gradient boosting algorithm optimized for tabular data.

### Model Parameters
- `n_estimators`: 1000
- `learning_rate`: 0.05
- `max_depth`: 6

### Training and Testing
- Split the dataset into training and testing sets.
- Trained the model on the training data and evaluated it on the test set.

---

## Evaluation
### Performance Metric
- **Root Mean Squared Error (RMSE)**: 145,813
  - The RMSE indicates the average error in predicted house prices.
  - The mean house price in the dataset: 609,736

---

## Results and Insights
1. The model performed reasonably well, with an RMSE of approximately 24% of the mean price.
2. The addition of time-based features improved the model’s ability to capture seasonal trends.
3. Outlier detection and removal contributed to better model stability and performance.

---

## Visualizations
### Actual vs Predicted Prices
The following graph visualizes the comparison between actual and predicted house prices over time:

![Actual vs Predicted Prices](https://github.com/Abdelrahman2030/house_sales_time_series/blob/main/Graphs/actual_vs_predicted.png)

---

## Conclusion
This project demonstrates the use of XGBoost and advanced preprocessing techniques to predict house sales prices. Future improvements could include:
- Experimenting with additional models, such as LightGBM or neural networks.
- Incorporating external data sources, such as economic indicators, to improve predictions.
- Optimizing hyperparameters using advanced techniques like Bayesian Optimization.

---

## Reproducibility
To reproduce this project:
1. Clone the repository or access the dataset.
2. Follow the preprocessing steps as described.
3. Train the XGBoost model with the provided parameters.
4. Evaluate the model on the test set using RMSE.

For detailed implementation, refer to the code provided in the repository.
