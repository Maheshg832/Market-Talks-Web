import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import os

@st.cache_data(ttl=1800, show_spinner=False)
def get_market_news(sources="bloomberg,cnbc,financial-times,business-insider,fortune", limit=10):
    """
    Fetch market news from NewsAPI
    
    Parameters:
    sources (str): Comma-separated list of news sources
    limit (int): Number of news articles to fetch
    
    Returns:
    list: List of news articles
    """
    try:
        # Get API key from environment variable
        api_key = os.getenv("NEWSAPI_KEY", "")
        
        # If no API key is provided, return mock data with a warning
        if not api_key:
            st.warning("NewsAPI key not provided. Using limited news data.")
            return get_mock_news()
        
        # Set up the request
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": api_key,
            "sources": sources,
            "pageSize": limit,
            "category": "business"
        }
        
        # Make the request
        response = requests.get(url, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            return articles
        else:
            st.error(f"Error fetching news: {response.status_code} - {response.text}")
            return get_mock_news()
    except Exception as e:
        st.error(f"Error fetching market news: {str(e)}")
        return get_mock_news()

@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_news(symbol, limit=5):
    """
    Fetch news for a specific stock
    
    Parameters:
    symbol (str): Stock symbol
    limit (int): Number of news articles to fetch
    
    Returns:
    list: List of news articles
    """
    try:
        # Get API key from environment variable
        api_key = os.getenv("NEWSAPI_KEY", "")
        
        # If no API key is provided, return mock data with a warning
        if not api_key:
            st.warning("NewsAPI key not provided. Using limited stock news data.")
            return get_mock_stock_news(symbol)
        
        # Get company name
        ticker = yf.Ticker(symbol)
        company_name = ticker.info.get('shortName', symbol)
        
        # Calculate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days
        
        # Format dates
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # Set up the request
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": api_key,
            "q": f"{company_name} OR {symbol}",
            "from": from_date,
            "to": to_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": limit
        }
        
        # Make the request
        response = requests.get(url, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            return articles
        else:
            st.error(f"Error fetching stock news: {response.status_code} - {response.text}")
            return get_mock_stock_news(symbol)
    except Exception as e:
        st.error(f"Error fetching stock news: {str(e)}")
        return get_mock_stock_news(symbol)

def get_mock_news():
    """
    Get mock news data when API is not available
    
    Returns:
    list: List of mock news articles
    """
    current_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    return [
        {
            "source": {"id": "financial-times", "name": "Financial Times"},
            "author": "Financial Times",
            "title": "Markets update: Global stocks show mixed performance",
            "description": "Global markets showed mixed performance as investors weigh economic data and central bank policies.",
            "url": "https://www.ft.com",
            "urlToImage": "",
            "publishedAt": current_date,
            "content": "Global markets showed mixed performance as investors weigh economic data and central bank policies."
        },
        {
            "source": {"id": "bloomberg", "name": "Bloomberg"},
            "author": "Bloomberg News",
            "title": "Fed signals potential rate adjustment as inflation data comes in",
            "description": "The Federal Reserve has indicated it may adjust rates depending on upcoming inflation data.",
            "url": "https://www.bloomberg.com",
            "urlToImage": "",
            "publishedAt": yesterday,
            "content": "The Federal Reserve has indicated it may adjust rates depending on upcoming inflation data."
        },
        {
            "source": {"id": "cnbc", "name": "CNBC"},
            "author": "CNBC Staff",
            "title": "Technology sector leads market gains amid AI optimism",
            "description": "Technology stocks continue to outperform as AI developments drive investor optimism.",
            "url": "https://www.cnbc.com",
            "urlToImage": "",
            "publishedAt": yesterday,
            "content": "Technology stocks continue to outperform as AI developments drive investor optimism."
        }
    ]

def get_mock_stock_news(symbol):
    """
    Get mock news data for a specific stock when API is not available
    
    Parameters:
    symbol (str): Stock symbol
    
    Returns:
    list: List of mock news articles for the stock
    """
    current_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    return [
        {
            "source": {"id": "financial-times", "name": "Financial Times"},
            "author": "Financial Times",
            "title": f"{symbol} reports quarterly earnings above expectations",
            "description": f"{symbol} has reported quarterly earnings that exceeded analyst expectations, driving stock price higher.",
            "url": "https://www.ft.com",
            "urlToImage": "",
            "publishedAt": current_date,
            "content": f"{symbol} has reported quarterly earnings that exceeded analyst expectations, driving stock price higher."
        },
        {
            "source": {"id": "bloomberg", "name": "Bloomberg"},
            "author": "Bloomberg News",
            "title": f"Analysts upgrade {symbol} on positive outlook",
            "description": f"Several analysts have upgraded {symbol} citing positive future outlook and growth potential.",
            "url": "https://www.bloomberg.com",
            "urlToImage": "",
            "publishedAt": yesterday,
            "content": f"Several analysts have upgraded {symbol} citing positive future outlook and growth potential."
        }
    ]
