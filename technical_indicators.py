import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_indicators(df):
    """
    Calculate technical indicators for a stock
    
    Parameters:
    df (pandas.DataFrame): DataFrame with stock data
    
    Returns:
    pandas.DataFrame: DataFrame with technical indicators
    """
    # Create a copy to avoid modifying the original DataFrame
    indicators = pd.DataFrame(index=df.index)
    
    # Calculate RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    indicators['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    
    indicators['MACD'] = ema12 - ema26
    indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
    indicators['MACD_Hist'] = indicators['MACD'] - indicators['MACD_Signal']
    
    # Calculate Bollinger Bands
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    
    indicators['BB_Upper'] = sma20 + (std20 * 2)
    indicators['BB_Middle'] = sma20
    indicators['BB_Lower'] = sma20 - (std20 * 2)
    
    # Calculate Moving Averages
    indicators['SMA_50'] = df['Close'].rolling(window=50).mean()
    indicators['SMA_200'] = df['Close'].rolling(window=200).mean()
    indicators['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Calculate Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    
    indicators['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    indicators['Stoch_D'] = indicators['Stoch_K'].rolling(window=3).mean()
    
    # Calculate Average Directional Index (ADX)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff(-1).abs()
    
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
    
    tr = pd.DataFrame({
        'tr1': (df['High'] - df['Low']).abs(),
        'tr2': (df['High'] - df['Close'].shift()).abs(),
        'tr3': (df['Low'] - df['Close'].shift()).abs()
    }).max(axis=1)
    
    atr = tr.rolling(window=14).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    indicators['ADX'] = dx.rolling(window=14).mean()
    
    # Calculate OBV (On-Balance Volume)
    obv = pd.Series(0, index=df.index)
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    indicators['OBV'] = obv
    
    return indicators

def identify_patterns(df):
    """
    Identify chart patterns
    
    Parameters:
    df (pandas.DataFrame): DataFrame with stock data
    
    Returns:
    dict: Dictionary with identified patterns
    """
    patterns = {}
    
    # Identify support and resistance levels
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    
    # Identify recent high and low points
    patterns['support'] = min(low[-20:])
    patterns['resistance'] = max(high[-20:])
    
    # Check for bullish engulfing pattern
    for i in range(1, len(df)-1):
        # Bullish engulfing
        if df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and \
           df['Close'].iloc[i] > df['Open'].iloc[i] and \
           df['Open'].iloc[i] <= df['Close'].iloc[i-1] and \
           df['Close'].iloc[i] > df['Open'].iloc[i-1]:
            patterns['bullish_engulfing'] = df.index[i]
    
    # Check for bearish engulfing pattern
    for i in range(1, len(df)-1):
        # Bearish engulfing
        if df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and \
           df['Close'].iloc[i] < df['Open'].iloc[i] and \
           df['Open'].iloc[i] >= df['Close'].iloc[i-1] and \
           df['Close'].iloc[i] < df['Open'].iloc[i-1]:
            patterns['bearish_engulfing'] = df.index[i]
    
    # Check for hammer pattern
    for i in range(1, len(df)-1):
        body_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
        lower_shadow = min(df['Open'].iloc[i], df['Close'].iloc[i]) - df['Low'].iloc[i]
        upper_shadow = df['High'].iloc[i] - max(df['Open'].iloc[i], df['Close'].iloc[i])
        
        if lower_shadow > 2 * body_size and upper_shadow < 0.1 * body_size:
            patterns['hammer'] = df.index[i]
    
    return patterns

def generate_signals_from_indicators(df, indicators):
    """
    Generate trading signals from technical indicators
    
    Parameters:
    df (pandas.DataFrame): DataFrame with stock data
    indicators (pandas.DataFrame): DataFrame with technical indicators
    
    Returns:
    pandas.DataFrame: DataFrame with trading signals
    """
    signals = pd.DataFrame(index=df.index)
    signals['Signal'] = None
    
    # RSI signals
    signals.loc[indicators['RSI'] < 30, 'RSI_Signal'] = 'Buy'
    signals.loc[indicators['RSI'] > 70, 'RSI_Signal'] = 'Sell'
    
    # MACD signals
    signals.loc[indicators['MACD'] > indicators['MACD_Signal'], 'MACD_Signal'] = 'Buy'
    signals.loc[indicators['MACD'] < indicators['MACD_Signal'], 'MACD_Signal'] = 'Sell'
    
    # Bollinger Bands signals
    signals.loc[df['Close'] < indicators['BB_Lower'], 'BB_Signal'] = 'Buy'
    signals.loc[df['Close'] > indicators['BB_Upper'], 'BB_Signal'] = 'Sell'
    
    # Moving Average signals
    signals.loc[indicators['SMA_50'] > indicators['SMA_200'], 'MA_Signal'] = 'Buy'
    signals.loc[indicators['SMA_50'] < indicators['SMA_200'], 'MA_Signal'] = 'Sell'
    
    # Combined signal
    signals['Buy_Count'] = signals[['RSI_Signal', 'MACD_Signal', 'BB_Signal', 'MA_Signal']].apply(
        lambda x: (x == 'Buy').sum(), axis=1
    )
    
    signals['Sell_Count'] = signals[['RSI_Signal', 'MACD_Signal', 'BB_Signal', 'MA_Signal']].apply(
        lambda x: (x == 'Sell').sum(), axis=1
    )
    
    signals.loc[signals['Buy_Count'] >= 2, 'Signal'] = 'Buy'
    signals.loc[signals['Sell_Count'] >= 2, 'Signal'] = 'Sell'
    
    # Fill forward to ensure we have signals
    signals['Signal'] = signals['Signal'].ffill()
    
    return signals.dropna(subset=['Signal'])
