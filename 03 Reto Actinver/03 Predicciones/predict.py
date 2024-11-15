import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest(data, model, predictors, start, step):
        all_predictions = []
        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            if test.empty:
                continue
            model.fit(train[predictors], train["Target"])
            preds = model.predict_proba(test[predictors])[:,1]
            preds[preds >= .6] = 1
            preds[preds < .6] = 0
            preds = pd.Series(preds, index=test.index, name="Predictions")
            combined = pd.concat([test["Target"], preds], axis=1)
            all_predictions.append(combined)
        if not all_predictions:
            raise ValueError("No predictions were made. Check the start and step parameters.")
        return pd.concat(all_predictions)

def prediction_tomorrow_1(stock_symbol):
    
    file_name = f"{stock_symbol}.csv"
    
    if os.path.exists(file_name):
        stock_data = pd.read_csv(file_name, index_col=0)
    else:
        stock = yf.Ticker(stock_symbol)
        stock_data = stock.history(period="max")
        stock_data.to_csv(file_name)

    stock_data.index = pd.to_datetime(stock_data.index, utc=True)
    
    # Get the start date from the maximum period
    start_date = stock_data.index.min()
    print(f"Start Date: {start_date}")

    # Filter the data to start from the start date
    stock_data = stock_data.loc[start_date:].copy()

    # Remove unwanted columns
    if "Dividends" in stock_data.columns:
        del stock_data["Dividends"]
    if "Stock Splits" in stock_data.columns:
        del stock_data["Stock Splits"]

    # Create Target column for prediction
    stock_data["Tomorrow"] = stock_data["Close"].shift(-1)
    stock_data["Target"] = (stock_data["Tomorrow"] > stock_data["Close"]).astype(int)

    # Add rolling averages and trends as new predictors
    horizons = [2, 5, 60, 250, 1000]
    predictors = ["Close", "Volume", "Open", "High", "Low"]

    for horizon in horizons:
        rolling_averages = stock_data.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        stock_data[ratio_column] = stock_data["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{horizon}"
        stock_data[trend_column] = stock_data.shift(1).rolling(horizon).sum()["Target"]

        predictors += [ratio_column, trend_column]

    # Add more technical indicators
    stock_data["RSI"] = compute_rsi(stock_data["Close"])
    predictors.append("RSI")

    # Drop missing data
    stock_data = stock_data.dropna(subset=stock_data.columns[stock_data.columns != "Tomorrow"])

    # Normalize the data
    scaler = StandardScaler()
    stock_data[predictors] = scaler.fit_transform(stock_data[predictors])

    # Define model and hyperparameter tuning
    model = RandomForestClassifier(random_state=1)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'min_samples_split': [10, 20, 50]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='precision')

    # Run backtest
    grid_search.fit(stock_data[predictors], stock_data["Target"])
    best_model = grid_search.best_estimator_
    predictions = backtest(stock_data, best_model, predictors, start=2500, step=250)

    # Predict for tomorrow
    last_row = stock_data.iloc[-1:]
    tomorrow_prediction = best_model.predict(last_row[predictors])[0]

    # Precision score
    precision = precision_score(predictions["Target"], predictions["Predictions"])

    print("Tomorrow's Prediction:", "Up" if tomorrow_prediction == 1 else "Down")
    print("Precision Score:", precision)

    # Save the results to a file
    prediction_folder = "predictions"
    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)

    today_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_path = os.path.join(prediction_folder, f"model 1 {stock_symbol} tomorrow {today_date}.txt")

    with open(file_path, "w") as file:
        file.write(f"Tomorrow's Prediction: {'Up' if tomorrow_prediction == 1 else 'Down'}\n")
        file.write(f"Precision Score: {precision}\n")

    
    return tomorrow_prediction, precision

    


def prediction_tomorrow_2(stock_symbol):
    # Load or fetch stock data
    file_name = f"{stock_symbol}.csv"

    if os.path.exists(file_name):
        stock_data = pd.read_csv(file_name, index_col=0)
    else:
        stock = yf.Ticker(stock_symbol)
        stock_data = stock.history(period="max")
        stock_data.to_csv(file_name)

    stock_data.index = pd.to_datetime(stock_data.index, utc=True)
    
    # Get the start date from the maximum period
    start_date = stock_data.index.min()
    print(f"Start Date: {start_date}")

    stock_data = stock_data.loc[start_date:].copy()

    # Remove unwanted columns
    if "Dividends" in stock_data.columns:
        del stock_data["Dividends"]
    if "Stock Splits" in stock_data.columns:
        del stock_data["Stock Splits"]

    # Create Target column for prediction
    stock_data["Tomorrow"] = stock_data["Close"].shift(-1)
    stock_data["Target"] = (stock_data["Tomorrow"] > stock_data["Close"]).astype(int)

    # Add rolling averages and trends as new predictors
    horizons = [2, 5, 60, 250, 1000]
    predictors = ["Close", "Volume", "Open", "High", "Low"]

    for horizon in horizons:
        rolling_averages = stock_data.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        stock_data[ratio_column] = stock_data["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{horizon}"
        stock_data[trend_column] = stock_data.shift(1).rolling(horizon).sum()["Target"]

        predictors += [ratio_column, trend_column]

    # Add more technical indicators
    stock_data["RSI"] = compute_rsi(stock_data["Close"])
    predictors.append("RSI")

    # Drop missing data
    stock_data = stock_data.dropna(subset=stock_data.columns[stock_data.columns != "Tomorrow"])

    # Normalize the data
    scaler = StandardScaler()
    stock_data[predictors] = scaler.fit_transform(stock_data[predictors])

    # Define model and hyperparameter tuning
    model = RandomForestClassifier(random_state=1)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'min_samples_split': [10, 20, 50]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='precision')

    # Run backtest
    grid_search.fit(stock_data[predictors], stock_data["Target"])
    best_model = grid_search.best_estimator_
    predictions = backtest(stock_data, best_model, predictors, start=500, step=100)

    # Predict for tomorrow
    last_row = stock_data.iloc[-1:]
    tomorrow_prediction = best_model.predict(last_row[predictors])[0]

    # Precision score
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    
    print("Tomorrow's Prediction:", "Up" if tomorrow_prediction == 1 else "Down")
    print("Precision Score:", precision)

    # Save the results to a file
    prediction_folder = "predictions"
    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)

    today_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_path = os.path.join(prediction_folder, f"model 2 {stock_symbol} tomorrow {today_date}.txt")

    with open(file_path, "w") as file:
        file.write(f"Tomorrow's Prediction: {'Up' if tomorrow_prediction == 1 else 'Down'}\n")
        file.write(f"Precision Score: {precision}\n")
    
    return tomorrow_prediction, precision


def prediction_five_days(stock_symbol):
    
    # Load or fetch stock data
    file_name = f"{stock_symbol}.csv"
    
    if os.path.exists(file_name):
        stock_data = pd.read_csv(file_name, index_col=0)
    else:
        stock = yf.Ticker(stock_symbol)
        stock_data = stock.history(period="max")
        stock_data.to_csv(file_name)

    stock_data.index = pd.to_datetime(stock_data.index, utc=True)
    
    # Get the start date from the maximum period
    start_date = stock_data.index.min()
    print(f"Start Date: {start_date}")

    stock_data = stock_data.loc[start_date:].copy()

    # Remove unwanted columns
    if "Dividends" in stock_data.columns:
        del stock_data["Dividends"]
    if "Stock Splits" in stock_data.columns:
        del stock_data["Stock Splits"]

    # Create Target column for prediction (next 5 days)
    stock_data["In5Days"] = stock_data["Close"].shift(-5)
    stock_data["Target"] = (stock_data["In5Days"] > stock_data["Close"]).astype(int)

    # Add rolling averages and trends as new predictors
    horizons = [2, 5, 60, 250, 1000]
    predictors = ["Close", "Volume", "Open", "High", "Low"]

    for horizon in horizons:
        rolling_averages = stock_data.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        stock_data[ratio_column] = stock_data["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{horizon}"
        stock_data[trend_column] = stock_data.shift(1).rolling(horizon).sum()["Target"]

        predictors += [ratio_column, trend_column]

    # Add more technical indicators
    stock_data["RSI"] = compute_rsi(stock_data["Close"])
    predictors.append("RSI")

    # Drop missing data
    stock_data = stock_data.dropna(subset=stock_data.columns[stock_data.columns != "In5Days"])

    # Normalize the data
    scaler = StandardScaler()
    stock_data[predictors] = scaler.fit_transform(stock_data[predictors])

    # Define model and hyperparameter tuning
    model = RandomForestClassifier(random_state=1)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'min_samples_split': [10, 20, 50]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='precision')

    # Run backtest
    grid_search.fit(stock_data[predictors], stock_data["Target"])
    best_model = grid_search.best_estimator_
    predictions = backtest(stock_data, best_model, predictors, start=2500, step=250)

    # Predict for the next 5 days
    last_row = stock_data.iloc[-1:]
    five_day_prediction = best_model.predict(last_row[predictors])[0]

    # Precision score
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    
    print("Prediction:", "Up" if five_day_prediction == 1 else "Down")
    print("Precision Score:", precision)

    # Save the results to a file
    prediction_folder = "predictions"
    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)

    today_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_path = os.path.join(prediction_folder, f"{stock_symbol} 5 days {today_date}.txt")

    with open(file_path, "w") as file:
        file.write(f"5 days Prediction: {'Up' if five_day_prediction == 1 else 'Down'}\n")
        file.write(f"Precision Score: {precision}\n")
    
    return five_day_prediction, precision