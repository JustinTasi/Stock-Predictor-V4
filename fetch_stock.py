import pandas as pd
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose

def download_and_prepare_data(ticker_symbol):
    # Function to download and prepare data
    print(f"Downloading data for {ticker_symbol} from Yahoo Finance...")
    
    # Download data using yfinance
    data = yf.download(ticker_symbol, period="max", interval="1d")
    
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    data_file = f"./data/{ticker_symbol}.csv"
    df.to_csv(data_file)
    
    print("Data downloaded and saved to", data_file)


# 因為yahoo是全世界的 要查指定台灣的要特別加 .TW
download_and_prepare_data('0050.TW')        
