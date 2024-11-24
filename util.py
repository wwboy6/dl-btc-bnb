import tensorflow as tf
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def fetchHistoricalData(symbol, start_date, end_date, interval='1d'):
    # Binance API endpoint for historical candlestick data
    url = "https://api.binance.com/api/v3/klines"
    
    # Set the parameters for the API request
    params = {
        'symbol': symbol,  # BNB against USDT
        'interval': interval,     # Daily interval
        'startTime': int(start_date.timestamp() * 1000),
        'endTime': int(end_date.timestamp() * 1000),
    }
    
    # Make the API request
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return None
    
    # Parse the response data
    data = response.json()
    
    # Create a DataFrame to store the data
    df = pd.DataFrame(data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    
    # Convert timestamps to readable dates
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # Select relevant columns
    df = df[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df

def plotPriceDatas(df, title):
  plt.figure(figsize=(14, 7))
  plt.plot(df['Open Time'], df['Open'], linestyle="-", label='Open', color='blue', alpha=0.6)
  plt.plot(df['Open Time'], df['High'], linestyle="-",label='High', color='green', alpha=0.6)
  plt.plot(df['Open Time'], df['Low'], linestyle="-",label='Low', color='red', alpha=0.6)
  plt.plot(df['Open Time'], df['Close'], linestyle="-",label='Close', color='orange', alpha=0.6)
  plt.title(title)
  plt.xlabel('Date')
  plt.ylabel('Price (USDT)')
  # plt.xticks(rotation=45)
  plt.legend()
  plt.grid()
  ax = plt.gca()
  ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

def prepareTrainingInputs(data, windowSize, testSize):
  dataSize = len(data)
  # count number of windows w.r.t. windowSize
  trainSize = dataSize - testSize - windowSize
  indexGenerators = [range(i, i+windowSize) for i in range(trainSize)]
  trainInputs = np.array([data[i] for i in indexGenerators])
  return trainInputs

def prepareTraingingOutputs(data, windowSize, testSize):
  dataSize = len(data)
  # count number of windows w.r.t. windowSize
  trainSize = dataSize - testSize - windowSize
  indexGenerator = range(windowSize, trainSize+windowSize)
  trainOutputs = np.array(data[indexGenerator])
  return trainOutputs

def prepareTestingInputs(data, windowSize, testSize):
  dataSize = len(data)
  indexGenerators = [range(dataSize - testSize - windowSize + i, dataSize - testSize + i) for i in range(testSize)]
  trainInputs = np.array([data[i] for i in indexGenerators])
  return trainInputs

def prepareTestingOutputs(data, testSize):
  dataSize = len(data)
  indexGenerator = range(dataSize - testSize, dataSize)
  testInputs = np.array(data[indexGenerator])
  return testInputs

def prepareDataSetFromArray(x_train, y_train, x_test, y_test, batch_size=1024):
  train_features_dataset = tf.data.Dataset.from_tensor_slices(x_train)
  train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)
  test_features_dataset = tf.data.Dataset.from_tensor_slices(x_test)
  test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

  train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
  test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

  train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
  test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  return train_dataset, test_dataset

def plotHistoryRSME(history, lastEpoch=150): 
  rmse = np.sqrt(history.history['val_mse'])
  plt.plot(rmse[-lastEpoch:-1], label="val rmse")
  plt.plot(np.sqrt(history.history['mse'])[-lastEpoch:-1], label="train rmse")
  plt.title("Root mean squared error in log scale")
  plt.legend()
