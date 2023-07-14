import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd
from transformers import pipeline

sp500 = yf.Ticker("^GSPC")  # S&P 500 data
data = sp500.history(period="max")  # get the entire hostory of the index
# data clean up we don't need dividends and stock splits values
del data['Dividends']
del data['Stock Splits']

# add a new column for tomorrow
data['Tomorrow'] = data['Close'].shift(-1)

# create the target that we want
data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)

data = data.loc['1990-01-01':].copy()

model = RandomForestClassifier(
    n_estimators=100, min_samples_split=100, random_state=1)

train = data.iloc[:-100]  # remove the last 100 values
test = data.iloc[:-100]  # remove the last 100 values

predictors = ["Close", "Volume", "Open", "High", "Low"]

model.fit(train[predictors], train['Target'])

predictions = model.predict(test[predictors])

predictions = pd.Series(predictions, index=test.index)
score = precision_score(test['Target'], predictions)
print(score)
