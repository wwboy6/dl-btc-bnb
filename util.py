import tensorflow as tf
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
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

def prepareDataSetFromArray(x_train, y_train, x_test, y_test, shuffle=False, batch_size=1024):
  train_features_dataset = tf.data.Dataset.from_tensor_slices(x_train)
  train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)
  test_features_dataset = tf.data.Dataset.from_tensor_slices(x_test)
  test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

  train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
  if shuffle:
    train_dataset = train_dataset.shuffle(buffer_size=1024)
  test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

  train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
  test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  return train_dataset, test_dataset

def standardTrainingAndReport(model, x_test, y_test, train_dataset, test_dataset, early_stopping_patience=20, early_stopping_restore_best_weights=True, reduce_lr_patience=5, plotHistoryLastEpoch=0):
  history = model.fit(train_dataset,
            epochs=5000, # just a large number
            validation_data=test_dataset,
            verbose=0, # prevent large amounts of training outputs
            callbacks=[
              tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=early_stopping_patience, restore_best_weights=early_stopping_restore_best_weights, verbose=1),
              tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=reduce_lr_patience, verbose=1)
            ])
  plotHistoryRSME(history, plotHistoryLastEpoch)
  loss = np.sqrt(model.evaluate(x_test, y_test)[0])
  print(f"loss: {loss}")
  # plot prediction
  prediction = model.predict(x_test)
  plt.figure(figsize=(12,5))
  plt.plot(prediction, label='Prediction')
  plt.plot(y_test, label='actual')
  plt.title('Prediction vs actual')
  plt.legend()
  corr = np.corrcoef(prediction.reshape(-1), y_test.reshape(-1))[0, 1]  
  print(f"corr: {corr}")
  return history, loss, corr

def noValidationTrainingAndReport(model, x_test, y_test, train_dataset, epochs=50,plotHistoryLastEpoch=0):
  history = model.fit(train_dataset,
            epochs=epochs,
            verbose=0, # prevent large amounts of training outputs
            )
  # plotHistoryRSME(history, plotHistoryLastEpoch)
  plt.figure(figsize=(12,5))
  plt.plot(np.sqrt(history.history['mse'])[-plotHistoryLastEpoch:-1], label="train rmse")
  plt.title("Root mean squared error in log scale")
  plt.legend()
  plt.yscale("log")
  #
  loss = np.sqrt(model.evaluate(x_test, y_test)[0])
  print(f"loss: {loss}")
  # plot prediction
  prediction = model.predict(x_test)
  plt.figure(figsize=(12,5))
  plt.plot(prediction, label='Prediction')
  plt.plot(y_test, label='actual')
  plt.title('Prediction vs actual')
  plt.legend()
  corr = np.corrcoef(prediction.reshape(-1), y_test.reshape(-1))[0, 1] 
  print(f"corr: {corr}")
  return history, loss, corr

def plotHistoryRSME(history, lastEpoch=0): 
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
  data = data.reshape(-1)
  colors = ['green' if value > 0 else 'red' for value in data]
  plt.bar(range(len(data)), data, color=colors)

def computeProfitOfFuture(data, future_period):
  return np.array([data[i+future_period] - data[i] for i in range(len(data)-future_period)])

# resample data pair so that the amount of positive y_train would be the same as negative
def undersampleSeriesDataTomekBySign(x, y):
  dataCount = len(y)
  assert(dataCount == len(x))
  # get the amount of positive and negative samples
  posDeterminer = lambda v: v > 0
  negDeterminer = lambda v: v < 0
  posCount = len(y[posDeterminer(y)])
  negCount = len(y[negDeterminer(y)])
  if (posCount == negCount):
    return x, y
  isFilteringPosCount = posCount > negCount
  if isFilteringPosCount:
    filteringDeterminer = posDeterminer
  else:
    filteringDeterminer = negDeterminer
  # undersample the data with determiner until the amount of positive and negative samples are the same
  lastIndex = 0
  while posCount != negCount:
    # get the index of the first element that should be filtered, and it is next to another index that should not be filtered
    index = next((i+lastIndex for i, yy in enumerate(y[lastIndex:]) if (filteringDeterminer(yy) and (
      (not filteringDeterminer(y[max(0, i+lastIndex-1)])) or (not filteringDeterminer(y[min(dataCount-1, i+lastIndex+1)]))
    ))), None)
    if index is None:
      if (lastIndex == 0):
        raise Exception("No element can be filtered")
      lastIndex = 0
      continue
    # remove the element at the index
    x = np.delete(x, index, axis=0)
    y = np.delete(y, index, axis=0)
    # update the counters
    if isFilteringPosCount:
      posCount -= 1
    else:
      negCount -= 1
    dataCount -= 1
    lastIndex = index
    # avoid removing same group again
    if lastIndex == 0 or not filteringDeterminer(y[lastIndex - 1]):
      lastIndex = (lastIndex + 1) % dataCount
  return x, y

def rescaleOneValueBySign(v, y_pos_scale, y_neg_scale):
  if v > 0:
    return v / (y_pos_scale)
  else:
    return v / (y_neg_scale)

def rescalingBySign(y):
  y_pos_mean = np.mean(y[y>0])
  y_neg_mean = -np.mean(y[y<0])
  print(f"y_pos_mean: {y_pos_mean}")
  print(f"y_neg_mean: {y_neg_mean}")
  scale_target = min(y_pos_mean, y_neg_mean)
  print(f"scale_target: {scale_target}")
  y_pos_scale = y_pos_mean / scale_target
  y_neg_scale = y_neg_mean / scale_target
  return np.array([rescaleOneValueBySign(v, y_pos_scale, y_neg_scale) for v in y])

def computeUDNTrend(data, ut_target, dt_target, nt_target, scaling):
  data_count = len(data)
  zeros = np.zeros(data_count)
  up_trend = 1-np.tanh(np.max([ut_target - data, zeros], axis=0) * scaling)
  down_trend = 1-np.tanh(np.max([data - dt_target, zeros], axis=0) * scaling)
  neutral_trend = 1 - np.tanh(np.max([np.abs(data) - nt_target, zeros], axis=0) * scaling)
  return up_trend, down_trend, neutral_trend

def colored_line(x, y, c, **lc_kwargs):
  # Default the capstyle to butt so that the line segments smoothly line up
  default_kwargs = {"capstyle": "butt"}
  default_kwargs.update(lc_kwargs)
  # Compute the midpoints of the line segments. Include the first and last points
  # twice so we don't need any special syntax later to handle them.
  x = np.asarray(x)
  y = np.asarray(y)
  x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
  y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))
  # Determine the start, middle, and end coordinate pair of each line segment.
  # Use the reshape to add an extra dimension so each pair of points is in its
  # own list. Then concatenate them to create:
  # [
  #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
  #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
  #   ...
  # ]
  coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
  coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
  coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
  segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)
  lc = LineCollection(segments, **default_kwargs)
  lc.set_array(c)  # set the colors of each segment
  return lc

def drawHorizontalColoredBar(fig, ax, data, yOffset, cmap, linewidth, label, norm=Normalize(vmin=0, vmax=1), drawColorBar=True):
  dataSize = len(data)
  x = [i for i in range(0, dataSize)]
  y = np.zeros(dataSize) + yOffset
  # Create a Normalize object to fix the color scale
  lines = colored_line(x, y, data, linewidth=linewidth, cmap=cmap, norm=norm)
  sm = ax.add_collection(lines)
  if drawColorBar:
    fig.colorbar(sm, label=label, orientation='vertical', fraction=0.02, pad=0.04, shrink=1)

def drawBarsForTrends(fig, ax, up_trend_train, down_trend_train, neutral_trend_train, ut_target, norm=Normalize(vmin=0, vmax=1), drawColorBar=True):
  colors = [
      (0, 1, 0, 0),    # Transparent (RGBA)
      (0, 1, 0, 0.5)   # Green with 50% transparency (RGBA)
  ]
  cmap = LinearSegmentedColormap.from_list("transparent_to_green", colors, N=256)
  drawHorizontalColoredBar(fig, ax, up_trend_train, ut_target, cmap, 80, 'up trend', norm=norm, drawColorBar=drawColorBar)
  colors = [
      (1, 0, 0, 0),    # Transparent (RGBA)
      (1, 0, 0, 0.5)   # Red with 50% transparency (RGBA)
  ]
  cmap = LinearSegmentedColormap.from_list("transparent_to_red", colors, N=256)
  drawHorizontalColoredBar(fig, ax, down_trend_train, -ut_target, cmap, 80, 'down trend', norm=norm, drawColorBar=drawColorBar)
  colors = [
      (1, 1, 0, 0),    # Transparent (RGBA)
      (1, 1, 0, 0.5)   # Yellow with 50% transparency (RGBA)
  ]
  cmap = LinearSegmentedColormap.from_list("transparent_to_yellow", colors, N=256)
  drawHorizontalColoredBar(fig, ax, neutral_trend_train, 0, cmap, 80, 'neutral', norm=norm, drawColorBar=drawColorBar)
