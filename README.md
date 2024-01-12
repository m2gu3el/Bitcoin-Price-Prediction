# Bitcoin-Price-Prediction With Diverse Model Approaches
## Problem Statement
Predicting Bitcoin prices is tough because the market can be very unpredictable. This project aims to create a powerful Time Series Forecasting model to tackle this challenge.
## Methodology
1. Data Collection: Gather Bitcoin price data, from sources like yfinance or coindesk, and organize it into a CSV file.
2. Data Preprocessing: Prepare the data for modeling by cleaning and organizing it.
3. Model Development: Create various models one at a time, each using a different approach.
4. Model Evaluation: Plot curves and evaluate each model's performance using metrics like MAE, RMSE, MAPE, MASE and MSE.
5. Model Comparison: Explore different models to identify the most effective one.

## Contents of the Notebook
1. Downloading and Reading the Dataset
2. Model 0- Naive Model.
3. Model 1- Horizon=1, Window=7.
4. Model 1d- Horizon=7, Window=30.
5. Model 2- Conv1D Model.
6. Model 3- LSTM Model.
7. Preparation of Multivariate Data
8. Model 4- Multivariate Data Model.
9. Preparation of Data and creating Nbeats custom layers
10. Model 5- Nbeats Algorithm Model

## Objectives
1. Model Comparison: Test and compare different models.
2. Evaluation Metrics: Assess models based on MAE, RMSE, MAPE, MSE and MASE.
3. Prediction Intervals: Create intervals for time series model forecasts.

## Libraries Used
1. TensorFlow
2. Pandas
3. Matplotlib
4. NumPy
5. Scikit-learn (Sklearn)

## Dataset 
Link- https://finance.yahoo.com/quote/BTC-USD/history/
Timeperiod - 2014-12-12 to 2023-12-12


