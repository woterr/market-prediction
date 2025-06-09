# Market Prediction
The main idea of this project was to determine whether a company's current stock would move up or down based on historical trends. This document explains why models perform poorly on this kind of dataset, despite genuine effort.


# Logistic Regression
A logical step was to use Logistic regression trained on `GOOGL` stock data from 2024–2025. The model is tasked to find an appropriate curve (sigmoid function) which essentially returns probabilities of whether the stock market would move up or down the next day. 


## Data

Data fed to this model is attained using `yfinance` - Python module. The stock features include:
`High`, `Low`, `Open`, `Close`, and `Volume`.

These features are later engineered to give
`daily_return`, `volatility`, `price_range`, `close_open_diff`, `volume_change`. 

Read data.py to see how these features have been engineered. 

All features were **mean-normalized** before being fed into the model.

## Prediction

After training the logistic regression model and optimizing weights (see `utils.py`), it was tested on unseen 2023 data.

A threshold (0.55) was used to classify predictions into:

-   `UP` (1)
    
-   `DOWN` (0)
    
**Accuracy on test data**: ~70%


## Problems encountered
- **Noise** - The data is too noisy, which is expected considering it is a stock market. A logistic model is generally bad at training with noisy data.
- **Simplicity** - The model is too simple. There is no complexity to accommodate for sudden changes in the data. 
- **IMPORTANT**: The stock market grows based on human sentiment, company growth, news sentiment, macroeconomics, earnings, etc. The model does not have such data. It is very expensive and impractical to retrieve this data, over the last `N` years. 
- **Inflation** - Even with normalization, relative value differences across years may still impact predictions.
- **Bias** - The stock market has an overall increase over the last 2 years, so the model is based towards predicting `1`.


## Fixes
To increase model complexity and handle noise better, I tried using a **Neural Network** (TensorFlow).

# Neural Network

A new model was created using TenserFlow library - A neural network with 2 hidden layers. 

The network used a ReLU activation layer of 128 units, followed by another ReLU activation layer of 64 units. The output layer is a sigmoid function.

## Predictions
The problem of not having high quality data still persists. However, the model was able to do a little better with added complexity and reached an accuracy of 80%. 


## New problems

- **Overfitting** - The model easily overfit and it started memorizing the patterns. Even though it had good accuracy in predicting, it was purely luck. An overfit model usually tends to do terribly on the testing dataset. 

## Lessons learned

We cannot expect an ML model to predict something you cannot. The world is changing rapidly. Any global crisis can take place at any point in time, which directly affects the stock market.

Take COVID-19 for example, regardless of how well a given company was doing, all stock markets crashed due to this global pandemic, and an ML model could obviously not predict that.

Problems like this require time series models like LSTM or Transformers. However, the model would still not predict very well. This problem statement is constructed in a way that it cannot have a real world solution. 

# Final thoughts

Predicting the stock market is difficult. Impossible even. But this project did not leave me empty handed, I learnt valuable lessons. 

- Working with real world data is difficult and requires a lot of cleanup. 
- Real world problem statements should have a meaning, and a solution that is practically possible. 

