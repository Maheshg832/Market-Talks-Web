import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

@st.cache_data(ttl=3600, show_spinner=False)
def generate_features(df):
    """
    Generate features for ML models
    
    Parameters:
    df (pandas.DataFrame): DataFrame with stock data
    
    Returns:
    pandas.DataFrame: DataFrame with features
    """
    # Create a copy to avoid modifying the original DataFrame
    features = pd.DataFrame(index=df.index)
    
    # Price and volume features
    features['Close'] = df['Close']
    features['Volume'] = df['Volume']
    
    # Returns
    features['Daily_Return'] = df['Close'].pct_change()
    features['Daily_Return_Lag1'] = features['Daily_Return'].shift(1)
    features['Daily_Return_Lag2'] = features['Daily_Return'].shift(2)
    features['Daily_Return_Lag3'] = features['Daily_Return'].shift(3)
    
    # Price momentum
    for period in [5, 10, 20, 50, 200]:
        features[f'Return_{period}d'] = df['Close'].pct_change(periods=period)
        features[f'MA_{period}d'] = df['Close'].rolling(window=period).mean()
    
    # Price location relative to moving averages
    for period in [5, 10, 20, 50, 200]:
        features[f'Price_to_MA_{period}'] = df['Close'] / features[f'MA_{period}d'] - 1
    
    # Volatility
    for period in [5, 10, 20, 50]:
        features[f'Volatility_{period}d'] = df['Close'].pct_change().rolling(window=period).std()
    
    # Volume features
    features['Volume_Change'] = df['Volume'].pct_change()
    features['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
    features['Volume_Ratio'] = df['Volume'] / features['Volume_MA10']
    
    # High-Low range
    features['HL_Pct'] = (df['High'] - df['Low']) / df['Low']
    features['HL_Pct_MA5'] = features['HL_Pct'].rolling(window=5).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    features['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    
    features['MACD'] = ema12 - ema26
    features['MACD_Signal'] = features['MACD'].ewm(span=9, adjust=False).mean()
    features['MACD_Hist'] = features['MACD'] - features['MACD_Signal']
    
    # Drop NaN values
    features = features.dropna()
    
    return features

def train_signal_model(features, window=60):
    """
    Train model to predict buy/sell signals
    
    Parameters:
    features (pandas.DataFrame): DataFrame with features
    window (int): Training window
    
    Returns:
    tuple: (model, scaler)
    """
    # Prepare training data
    X = features.drop(columns=['Close', 'Volume']).iloc[:-window]
    
    # Create target: 1 if price goes up in the next 5 days, 0 otherwise
    future_return = features['Close'].pct_change(5).shift(-5)
    y = (future_return > 0).astype(int).iloc[:-window]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

def train_prediction_model(features, window=60):
    """
    Train model to predict price movements
    
    Parameters:
    features (pandas.DataFrame): DataFrame with features
    window (int): Training window
    
    Returns:
    tuple: (model, scaler)
    """
    # Prepare training data
    X = features.drop(columns=['Close', 'Volume']).iloc[:-window]
    
    # Create target: 5-day future return
    y = features['Close'].pct_change(5).shift(-5).iloc[:-window] * 100  # Convert to percentage
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

@st.cache_data(ttl=3600, show_spinner=False)
def generate_signals(df):
    """
    Generate AI-based trading signals
    
    Parameters:
    df (pandas.DataFrame): DataFrame with stock data
    
    Returns:
    pandas.DataFrame: DataFrame with signals
    """
    # Generate features
    features = generate_features(df)
    
    if len(features) < 200:
        # Not enough data for reliable predictions
        return pd.DataFrame()
    
    # Train models
    signal_model, signal_scaler = train_signal_model(features)
    prediction_model, prediction_scaler = train_prediction_model(features)
    
    # Prepare data for prediction
    X_latest = features.drop(columns=['Close', 'Volume'])
    X_scaled = signal_scaler.transform(X_latest)
    
    # Generate signals
    signal_proba = signal_model.predict_proba(X_scaled)[:, 1]  # Probability of class 1 (price goes up)
    predicted_move = prediction_model.predict(prediction_scaler.transform(X_latest))
    
    # Create signals DataFrame
    signals = pd.DataFrame(index=X_latest.index)
    signals['Confidence'] = signal_proba * 100  # Convert to percentage
    signals['Predicted_Move'] = predicted_move
    
    # Determine buy/sell signals
    signals['Signal'] = 'Hold'
    signals.loc[signals['Confidence'] > 60, 'Signal'] = 'Buy'
    signals.loc[signals['Confidence'] < 40, 'Signal'] = 'Sell'
    
    # Calculate actual move (for historical comparison)
    actual_returns = df['Close'].pct_change(5).shift(-5) * 100
    signals['Actual_Move'] = actual_returns
    
    # Display only strong signals
    signals = signals[(signals['Signal'] == 'Buy') | (signals['Signal'] == 'Sell')]
    
    # Ensure we have some signals
    if signals.empty:
        # Generate at least some recent signals based on thresholds
        recent_data = X_latest.iloc[-20:]
        recent_proba = signal_proba[-20:]
        recent_move = predicted_move[-20:]
        recent_actual = actual_returns.iloc[-20:]
        
        recent_signals = pd.DataFrame(index=recent_data.index)
        recent_signals['Confidence'] = recent_proba * 100
        recent_signals['Predicted_Move'] = recent_move
        recent_signals['Actual_Move'] = recent_actual
        
        # Determine signals
        recent_signals['Signal'] = 'Hold'
        recent_signals.loc[recent_signals['Confidence'] > 55, 'Signal'] = 'Buy'
        recent_signals.loc[recent_signals['Confidence'] < 45, 'Signal'] = 'Sell'
        
        # Filter and return
        return recent_signals[(recent_signals['Signal'] == 'Buy') | (recent_signals['Signal'] == 'Sell')]
    
    return signals
