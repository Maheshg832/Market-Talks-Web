import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(symbol, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance
    
    Parameters:
    symbol (str): Stock symbol
    start_date (datetime): Start date for data
    end_date (datetime): End date for data
    
    Returns:
    pandas.DataFrame: DataFrame with stock data
    """
    try:
        # Add one day to end_date to include the end date in the results
        end_date_adjusted = end_date + timedelta(days=1)
        
        # Fetch data
        data = yf.download(
            symbol,
            start=start_date,
            end=end_date_adjusted,
            progress=False
        )
        
        # Check if data is empty
        if data.empty:
            return None
            
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_info(symbol):
    """
    Fetch stock information and company details
    
    Parameters:
    symbol (str): Stock symbol
    
    Returns:
    dict: Dictionary with stock information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract relevant information
        stock_info = {
            'name': info.get('shortName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            'website': info.get('website', 'N/A'),
            'business_summary': info.get('longBusinessSummary', 'N/A')
        }
        
        return stock_info
    except Exception as e:
        st.error(f"Error fetching information for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_multiple_stocks_data(symbols, start_date, end_date):
    """
    Fetch data for multiple stocks
    
    Parameters:
    symbols (list): List of stock symbols
    start_date (datetime): Start date for data
    end_date (datetime): End date for data
    
    Returns:
    dict: Dictionary with DataFrames for each symbol
    """
    data = {}
    for symbol in symbols:
        df = get_stock_data(symbol, start_date, end_date)
        if df is not None:
            data[symbol] = df
    
    return data
