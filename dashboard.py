import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.data_fetcher import get_multiple_stocks_data
from utils.technical_indicators import calculate_indicators
from utils.ml_models import generate_signals
from utils.news_fetcher import get_market_news

st.set_page_config(
    page_title="Market Dashboard - Stock Analysis Platform",
    page_icon="📊",
    layout="wide"
)

# Session state initialization
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = True

if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Sidebar
with st.sidebar:
    st.title("Market Dashboard")
    
    # Dark mode toggle
    st.session_state['dark_mode'] = st.toggle("Dark Mode", value=st.session_state['dark_mode'])
    
    # Date range selection
    today = datetime.now()
    start_date = st.date_input("Start Date", today - timedelta(days=180))
    end_date = st.date_input("End Date", today)
    
    # Update button
    update_dashboard = st.button("Update Dashboard")

# Main content
st.title("Market Dashboard")

# Market overview section
st.header("Market Overview")

# Fetch data for major indices
indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']
indices_names = {
    '^GSPC': 'S&P 500',
    '^DJI': 'Dow Jones',
    '^IXIC': 'NASDAQ',
    '^RUT': 'Russell 2000'
}

try:
    # Fetch data
    indices_data = get_multiple_stocks_data(indices, start_date, end_date)
    
    if indices_data:
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        for i, (symbol, data) in enumerate(indices_data.items()):
            if data is not None and not data.empty:
                current_price = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2]
                price_change = current_price - prev_close
                pct_change = (price_change / prev_close) * 100
                
                col = [col1, col2, col3, col4][i]
                col.metric(
                    label=indices_names.get(symbol, symbol), 
                    value=f"{current_price:.2f}", 
                    delta=f"{pct_change:.2f}%"
                )
        
        # Create index performance chart
        st.subheader("Index Performance (normalized)")
        
        fig = go.Figure()
        
        for symbol, data in indices_data.items():
            if data is not None and not data.empty:
                # Normalize to 100
                normalized = (data['Close'] / data['Close'].iloc[0]) * 100
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=normalized,
                    mode='lines',
                    name=indices_names.get(symbol, symbol)
                ))
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Normalized Price (Base 100)',
            template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to fetch market index data. Please try again later.")
except Exception as e:
    st.error(f"Error in market overview: {str(e)}")

# Watchlist performance
st.header("Watchlist Performance")

try:
    # Fetch data for watchlist
    watchlist_data = get_multiple_stocks_data(st.session_state['watchlist'], start_date, end_date)
    
    if watchlist_data:
        # Create watchlist metrics
        watchlist_cols = st.columns(len(st.session_state['watchlist']))
        
        for i, (symbol, data) in enumerate(watchlist_data.items()):
            if data is not None and not data.empty:
                current_price = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2]
                price_change = current_price - prev_close
                pct_change = (price_change / prev_close) * 100
                
                col = watchlist_cols[i]
                col.metric(
                    label=symbol, 
                    value=f"${current_price:.2f}", 
                    delta=f"{pct_change:.2f}%"
                )
        
        # Create watchlist performance chart
        st.subheader("Watchlist Performance (normalized)")
        
        fig = go.Figure()
        
        for symbol, data in watchlist_data.items():
            if data is not None and not data.empty:
                # Normalize to 100
                normalized = (data['Close'] / data['Close'].iloc[0]) * 100
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=normalized,
                    mode='lines',
                    name=symbol
                ))
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Normalized Price (Base 100)',
            template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to fetch watchlist data. Please try again later.")
except Exception as e:
    st.error(f"Error in watchlist performance: {str(e)}")

# Trading signals section
st.header("AI Trading Signals")

try:
    # Create signal table for watchlist
    signals_data = []
    
    for symbol, data in watchlist_data.items():
        if data is not None and not data.empty:
            signals = generate_signals(data)
            
            if not signals.empty:
                latest_signal = signals.iloc[-1]
                
                signals_data.append({
                    'Symbol': symbol,
                    'Signal': latest_signal['Signal'],
                    'Confidence': f"{latest_signal['Confidence']:.1f}%",
                    'Predicted_Move': f"{latest_signal['Predicted_Move']:.2f}%",
                    'Last_Updated': signals.index[-1].strftime('%Y-%m-%d')
                })
    
    if signals_data:
        # Convert to DataFrame
        signals_df = pd.DataFrame(signals_data)
        
        # Apply styling
        def highlight_signals(val):
            if val == 'Buy':
                return 'background-color: darkgreen'
            elif val == 'Sell':
                return 'background-color: darkred'
            return ''
        
        styled_df = signals_df.style.applymap(highlight_signals, subset=['Signal'])
        
        # Display table
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No trading signals available for the current watchlist.")
except Exception as e:
    st.error(f"Error in trading signals: {str(e)}")

# Market news section
st.header("Market News")

try:
    # Fetch market news
    news = get_market_news(limit=6)
    
    if news:
        # Display news in expandable format
        for article in news:
            with st.expander(article['title']):
                st.write(f"**Source:** {article['source']['name']}")
                st.write(f"**Published:** {article['publishedAt'][:10]}")
                st.write(article['description'])
                st.write(f"[Read more]({article['url']})")
    else:
        st.info("No market news available at the moment.")
except Exception as e:
    st.error(f"Error fetching market news: {str(e)}")

# Heat map section
st.header("Sector Performance Heat Map")

try:
    # Define sectors and representative stocks
    sectors = {
        'Technology': ['AAPL', 'MSFT', 'NVDA', 'ADBE', 'CRM'],
        'Communication': ['GOOGL', 'META', 'NFLX', 'TWTR', 'DIS'],
        'Consumer Cyclical': ['AMZN', 'TSLA', 'NKE', 'SBUX', 'HD'],
        'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
        'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'UNH']
    }
    
    # Fetch data for sector stocks
    sector_returns = {}
    
    for sector_name, symbols in sectors.items():
        sector_data = get_multiple_stocks_data(symbols, start_date, end_date)
        
        # Calculate returns for each stock
        returns = []
        for symbol, data in sector_data.items():
            if data is not None and not data.empty:
                # Calculate 1-month return
                one_month_return = (data['Close'].iloc[-1] / data['Close'].iloc[-min(21, len(data))] - 1) * 100
                returns.append(one_month_return)
        
        # Calculate average sector return
        if returns:
            sector_returns[sector_name] = sum(returns) / len(returns)
    
    if sector_returns:
        # Create heat map
        fig = go.Figure()
        
        # Sort sectors by return
        sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
        
        # Create bars
        fig.add_trace(go.Bar(
            x=[s[0] for s in sorted_sectors],
            y=[s[1] for s in sorted_sectors],
            marker=dict(
                color=[s[1] for s in sorted_sectors],
                colorscale=['red', 'orange', 'green'],
                cmin=-5,
                cmax=5
            )
        ))
        
        fig.update_layout(
            title='One-Month Sector Performance (%)',
            template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sector performance data unavailable.")
except Exception as e:
    st.error(f"Error in sector performance: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Stock Market Analysis Platform | Data powered by Yahoo Finance")
