# Laptop Price Prediction

## Project Overview

This project aims to predict the price of laptops based on various features such as processor, RAM, storage, and brand using machine learning. It utilizes a Support Vector Regression (SVR) model to achieve this prediction.

## Code Structure and Logic

1. **Data Loading and Preprocessing:**
   - The code starts by loading the laptop price dataset from a CSV file using Pandas.
   - It checks for missing values and removes duplicate entries to ensure data quality.
   - Categorical features are encoded using Label Encoding to convert them into numerical representations.
   - Features are standardized using StandardScaler to bring them to a common scale.

2. **Model Training and Hyperparameter Tuning:**
   - The dataset is split into training and testing sets using `train_test_split`.
   - A Support Vector Regression (SVR) model is employed for price prediction.
   - Hyperparameter tuning is performed using `RandomizedSearchCV` to find the optimal model parameters.
   - The best model is selected based on the lowest Mean Squared Error (MSE).

3. **Prediction and Evaluation:**
   - The trained model is used to predict the prices of laptops in the test set.
   - The model's performance is evaluated using metrics such as Mean Squared Error (MSE), R-squared score, and Root Mean Squared Error (RMSE).
   - A scatter plot is generated to visualize the actual vs. predicted prices.
  
   ![image](https://github.com/user-attachments/assets/adb3a573-370a-4aac-a4fe-d526e1439782)


## Technology and Algorithms

- **Python:** The primary programming language used for this project.
- **Pandas:** Used for data manipulation and analysis.
- **NumPy:** Used for numerical computations.
- **Scikit-learn:** Used for machine learning tasks, including model training, evaluation, and preprocessing.
- **Support Vector Regression (SVR):** The machine learning algorithm used for price prediction. SVR is a powerful technique for regression tasks, especially when dealing with non-linear relationships between features and target variables.
- **Hyperparameter Tuning:** Techniques like `RandomizedSearchCV` are used to find the best settings for the SVR model, optimizing its performance.

## Usage

To run this project:

1.  Install the necessary libraries: `pandas`, `numpy`, `scikit-learn`, and `matplotlib`.
2.  Upload the `laptopPrice.csv` dataset to your Colab environment.
3.  Execute the code cells in the notebook sequentially.

## Conclusion

This project demonstrates the application of machine learning for predicting laptop prices. The use of SVR with hyperparameter tuning results in a model that can provide accurate price estimations.
