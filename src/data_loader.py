import logging
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple

# Logger configuration
logger = logging.getLogger(__name__)

def get_and_preprocess_data(
    tickers: List[str], 
    start_date: str = "1990-01-01", 
    end_date: str = "2025-01-01"
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Downloads historical data, calculates 8 predictive characteristics, 
    aligns the forward-looking target, and applies cross-sectional standardization.

    Args:
        tickers (List[str]): List of stock tickers to download.
        start_date (str): Start date for yfinance download.
        end_date (str): End date for yfinance download.

    Returns:
        Tuple[pd.DataFrame, List[str], List[str]]: 
            - Processed panel DataFrame.
            - List of feature column names.
            - List of unique tickers.
    """
    logger.info(f"Downloading data from yfinance ({start_date} to {end_date})...")
    
    # 1. Download Data
    data = yf.download(tickers, start=start_date, end=end_date, interval="1mo", progress=False)

    adj_close = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    volume = data['Volume']

    # 2. Transform into Panel (Long) Format
    df_close = adj_close.stack().reset_index()
    df_close.columns = ['Date', 'Ticker', 'Price']
    
    df_volume = volume.stack().reset_index()
    df_volume.columns = ['Date', 'Ticker', 'Volume']

    df = pd.merge(df_close, df_volume, on=['Date', 'Ticker'])
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    # 3. Calculate Target and Advanced Features
    logger.info("Calculating target and advanced features (preventing look-ahead bias)...")
    df['Return'] = df.groupby('Ticker')['Price'].pct_change()
    df['Target_R_t+1'] = df.groupby('Ticker')['Return'].shift(-1)

    df['Char_1_Reversal'] = df['Return']
    
    df['Char_2_Mom_12m'] = df.groupby('Ticker').apply(
        lambda x: (x['Price'].shift(1) / x['Price'].shift(12)) - 1
    ).reset_index(level=0, drop=True)
    
    df['Char_3_Mom_6m'] = df.groupby('Ticker').apply(
        lambda x: (x['Price'].shift(1) / x['Price'].shift(6)) - 1
    ).reset_index(level=0, drop=True)
    
    df['Char_4_Volatility'] = df.groupby('Ticker')['Return'].transform(
        lambda x: x.rolling(window=12, min_periods=6).std()
    )
    
    df['Char_5_LogVolume'] = np.log(df['Volume'].replace(0, np.nan))
    
    df['Char_6_Amihud'] = df['Return'].abs() / (df['Volume'] + 1e-8)
    
    df['Char_7_LT_Rev'] = df.groupby('Ticker').apply(
        lambda x: (x['Price'].shift(13) / x['Price'].shift(36)) - 1
    ).reset_index(level=0, drop=True)
    
    df['Char_8_MaxRet'] = df.groupby('Ticker')['Return'].transform(
        lambda x: x.rolling(window=12, min_periods=6).max()
    )

    df = df.dropna().reset_index(drop=True)

    # 4. Cross-Sectional Standardization
    logger.info("Applying cross-sectional standardization at each time step...")
    
    features = [
        'Char_1_Reversal', 'Char_2_Mom_12m', 'Char_3_Mom_6m', 'Char_4_Volatility', 
        'Char_5_LogVolume', 'Char_6_Amihud', 'Char_7_LT_Rev', 'Char_8_MaxRet'
    ]

    def cross_sectional_scaler(x: pd.Series) -> pd.Series:
        return (x - x.mean()) / (x.std() + 1e-8)

    df[features] = df.groupby('Date')[features].transform(cross_sectional_scaler)
    df[features] = df[features].fillna(0)
    
    logger.info(f"Data preprocessing complete! Final dataset shape: {df.shape}")
    return df, features, tickers
