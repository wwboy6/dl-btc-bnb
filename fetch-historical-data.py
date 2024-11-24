import sys
import requests
import pandas as pd

def fetchHistoricalData(symbol, start_date, end_date, interval='1d'):
  # Binance API endpoint for historical candlestick data
  url = "https://api.binance.com/api/v3/klines"
  data = []

  start_timestamp = int(start_date.timestamp() * 1000)
  end_timestamp = int(end_date.timestamp() * 1000)

  print(f"start_timestamp: {start_timestamp}")
  print(f"end_timestamp: {end_timestamp}")
  
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
      print("no data")
      break
    print(f"Fetched {len(sub_data)} sub-rows")
    data.extend(sub_data)
    # data may not be fully fetched
    # check the last start date and fetch it again from next day
    start_date = pd.to_datetime(sub_data[-1]['Open Time'])
    # increate start_date by 1 day
    start_date += pd.Timedelta(days=1)
    start_timestamp = int(start_date.timestamp() * 1000)
  
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

# get query label from command line arugment
if len(sys.argv) > 1:
    query_label = sys.argv[1]
    start_date_input = sys.argv[2]
    end_date_input = sys.argv[3]

# query_label = 'BNBUSDT'
# start_date_input = '2021-11-01'
# end_date_input = '2024-11-01'

start_date = pd.to_datetime(start_date_input)
end_date = pd.to_datetime(end_date_input)

bnb_data = fetchHistoricalData(query_label, start_date, end_date)

# save dataframe to csv file
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')
bnb_data.to_csv(f'data/{query_label}-{start_date_str}-{end_date_str}.csv', index=False)
