import tensorflow as tf
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def fetchHistoricalData(symbol, start_date, end_date, interval='1d'):
  # Binance API endpoint for historical candlestick data
  url = "https://api.binance.com/api/v3/klines"
  data = []

  start_timestamp = int(start_date.timestamp() * 1000)
  end_timestamp = int(end_date.timestamp() * 1000)
  
  while start_timestamp < end_timestamp:
    # Set the parameters for the API request
    params = {
      'symbol': symbol,  # BNB against USDT
      'interval': interval,     # Daily interval
      'startTime': start_timestamp,
      'endTime': end_timestamp,
    }
    
    # Make the API request
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
      print(f"Error fetching data: {response.status_code}")
      return None
    
    # Parse the response data
    sub_data = response.json()
    if not data:
      break
    print(f"Fetched {len(sub_data)} rows")
    data.extend(sub_data)
    # data may not be fully fetched
    # check the last start date and fetch it again from next day
    start_date = pd.to_datetime(sub_data[-1]['Open Time'])
    # increate start_date by 1 day
    start_date += pd.Timedelta(days=1)
    start_timestamp = start_date.timestamp() * 1000
  
  print(f"Fetched {len(data)} rows")

  # Create a DataFrame to store the data
  df = pd.DataFrame(data, columns=[
    'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume'
  ])
  
  # Convert timestamps to readable dates
  df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
  # Convert string to numeric
  df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
  df['High'] = pd.to_numeric(df['High'], errors='coerce')
  df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
  df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

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
  ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
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

def prepareDatasForSeriesModelTraining(inputData, outputData, windowSize, testSize, batch_size=1024):
  x_train = prepareTrainingInputs(inputData, windowSize, testSize)
  y_train = prepareTraingingOutputs(outputData, windowSize, testSize)
  x_test = prepareTestingInputs(inputData, windowSize, testSize)
  y_test = prepareTestingOutputs(outputData, testSize)
  train_dataset, test_dataset = prepareDataSetFromArray(x_train, y_train, x_test, y_test, batch_size)
  return x_train, y_train, x_test, y_test, train_dataset, test_dataset

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

def standardTrainingAndReport(model, x_test, y_test, train_dataset, test_dataset):
  model.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(0.001),
                metrics=["mae", "mse"])
  history = model.fit(train_dataset,
            epochs=5000, # just a large number
            validation_data=test_dataset,
            verbose=0, # prevent large amounts of training outputs
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)])
  plotHistoryRSME(history)
  rmse = np.sqrt(model.evaluate(x_test, y_test)[2])
  print(f"rmse: {rmse}")
  # plot prediction
  prediction = model.predict(x_test)
  plt.figure(figsize=(12,5))
  plt.plot(prediction, label='Prediction')
  plt.plot(y_test, label='actual')
  plt.title('Prediction vs actual')
  plt.legend()
  corr = np.corrcoef(prediction.reshape(-1), y_test)[0, 1] 
  print(f"corr: {corr}")
  return history, rmse, corr


def plotHistoryRSME(history, lastEpoch=150): 
  rmse = np.sqrt(history.history['val_mse'])
  plt.figure(figsize=(12,5))
  plt.plot(rmse[-lastEpoch:-1], label="val rmse")
  plt.plot(np.sqrt(history.history['mse'])[-lastEpoch:-1], label="train rmse")
  plt.title("Root mean squared error in log scale")
  plt.legend()
  plt.yscale("log")

def covertToLogScale(data):
  return np.array([np.log(abs(v) + 1)*np.sign(v) for v in data])

def plotBarColoredSign(data):
  colors = ['green' if value > 0 else 'red' for value in data]
  plt.bar(range(len(data)), data, color=colors)
