import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.data_fetcher import get_stock_data
from utils.technical_indicators import calculate_indicators
from utils.ml_models import generate_signals

# Page configuration
st.set_page_config(
    page_title="Stock Market Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = True

if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Sidebar
with st.sidebar:
    st.title("Stock Market Analysis")
    
    # Dark mode toggle
    st.session_state['dark_mode'] = st.toggle("Dark Mode", value=st.session_state['dark_mode'])
    
    # Stock search
    stock_symbol = st.text_input("Search Stock Symbol", "AAPL").upper()
    
    # Date range selection
    today = datetime.now()
    start_date = st.date_input("Start Date", today - timedelta(days=365))
    end_date = st.date_input("End Date", today)
    
    # Add to watchlist
    if st.button("Add to Watchlist"):
        if stock_symbol and stock_symbol not in st.session_state['watchlist']:
            st.session_state['watchlist'].append(stock_symbol)
            st.success(f"Added {stock_symbol} to watchlist!")
    
    # Watchlist display
    st.subheader("Watchlist")
    for i, symbol in enumerate(st.session_state['watchlist']):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"{symbol}")
        with col2:
            if st.button("X", key=f"remove_{i}"):
                st.session_state['watchlist'].remove(symbol)
                st.rerun()

# Main content
st.title("Stock Market Analysis Platform")

try:
    # Fetch stock data
    df = get_stock_data(stock_symbol, start_date, end_date)
    
    if df is not None and not df.empty:
        # Current price and daily change
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        price_change = current_price - prev_close
        pct_change = (price_change / prev_close) * 100
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric(
            label=f"{stock_symbol} Price", 
            value=f"${current_price:.2f}", 
            delta=f"{price_change:.2f} ({pct_change:.2f}%)"
        )
        
        col2.metric(
            label="Volume", 
            value=f"{df['Volume'].iloc[-1]:,}"
        )
        
        col3.metric(
            label="52-Week Range", 
            value=f"${df['Low'].min():.2f} - ${df['High'].max():.2f}"
        )
        
        # Stock price chart
        st.subheader(f"{stock_symbol} Stock Price")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlestick'
        ))
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            xaxis_rangeslider_visible=False,
            template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
            height=500,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators
        st.subheader("Technical Indicators")
        
        indicators = calculate_indicators(df)
        
        tab1, tab2, tab3 = st.tabs(["RSI", "MACD", "Bollinger Bands"])
        
        with tab1:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=indicators.index,
                y=indicators['RSI'],
                mode='lines',
                name='RSI'
            ))
            fig_rsi.add_hline(y=70, line_width=1, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_width=1, line_dash="dash", line_color="green")
            
            fig_rsi.update_layout(
                xaxis_title='Date',
                yaxis_title='RSI',
                template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig_rsi, use_container_width=True)
            st.write("""
            **Relative Strength Index (RSI)** is a momentum oscillator that measures the speed and change of price movements. 
            RSI oscillates between 0 and 100. Traditionally, RSI is considered overbought when above 70 and oversold when below 30.
            """)
        
        with tab2:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=indicators.index,
                y=indicators['MACD'],
                mode='lines',
                name='MACD'
            ))
            fig_macd.add_trace(go.Scatter(
                x=indicators.index,
                y=indicators['MACD_Signal'],
                mode='lines',
                name='Signal Line'
            ))
            fig_macd.add_trace(go.Bar(
                x=indicators.index,
                y=indicators['MACD_Hist'],
                name='Histogram'
            ))
            
            fig_macd.update_layout(
                xaxis_title='Date',
                yaxis_title='MACD',
                template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig_macd, use_container_width=True)
            st.write("""
            **Moving Average Convergence Divergence (MACD)** is a trend-following momentum indicator that shows the relationship
            between two moving averages of a security's price. The MACD is calculated by subtracting the 26-period Exponential Moving 
            Average (EMA) from the 12-period EMA. A 9-day EMA of the MACD, called the "signal line", is then plotted on top of the MACD.
            """)
        
        with tab3:
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(
                x=indicators.index,
                y=df['Close'],
                mode='lines',
                name='Close'
            ))
            fig_bb.add_trace(go.Scatter(
                x=indicators.index,
                y=indicators['BB_Upper'],
                mode='lines',
                name='Upper Band',
                line=dict(dash='dash')
            ))
            fig_bb.add_trace(go.Scatter(
                x=indicators.index,
                y=indicators['BB_Middle'],
                mode='lines',
                name='Middle Band',
                line=dict(dash='dot')
            ))
            fig_bb.add_trace(go.Scatter(
                x=indicators.index,
                y=indicators['BB_Lower'],
                mode='lines',
                name='Lower Band',
                line=dict(dash='dash')
            ))
            
            fig_bb.update_layout(
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig_bb, use_container_width=True)
            st.write("""
            **Bollinger Bands** are a type of statistical chart characterizing the prices and volatility over time of a financial 
            instrument. Bollinger Bands consist of a middle band being an N-period simple moving average (SMA), an upper band at 
            K times an N-period standard deviation above the middle band, and a lower band at K times an N-period standard deviation 
            below the middle band.
            """)
        
        # AI-based trading signals
        st.subheader("AI-Based Trading Signals")
        
        signals_df = generate_signals(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Most recent signal
            latest_signal = signals_df.iloc[-1]
            signal_color = "green" if latest_signal['Signal'] == 'Buy' else "red"
            
            st.markdown(f"<h3 style='color: {signal_color};'>Current Signal: {latest_signal['Signal']}</h3>", unsafe_allow_html=True)
            st.markdown(f"**Confidence**: {latest_signal['Confidence']:.2f}%")
            st.markdown(f"**Predicted Move**: {latest_signal['Predicted_Move']:.2f}%")
        
        with col2:
            # Historical signal accuracy
            accuracy = (signals_df['Actual_Move'] * signals_df['Predicted_Move'] > 0).mean() * 100
            st.metric("Signal Accuracy (Historical)", f"{accuracy:.1f}%")
            
            # Average return when following signals
            avg_return = signals_df.loc[signals_df['Signal'] == 'Buy', 'Actual_Move'].mean()
            st.metric("Average Return on Buy Signals", f"{avg_return:.2f}%")
        
        # Signal history chart
        fig_signals = go.Figure()
        
        # Add close price
        fig_signals.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Close Price'
        ))
        
        # Add buy signals
        buy_signals = signals_df[signals_df['Signal'] == 'Buy']
        fig_signals.add_trace(go.Scatter(
            x=buy_signals.index,
            y=df.loc[buy_signals.index, 'Close'],
            mode='markers',
            marker=dict(
                size=10,
                symbol='triangle-up',
                color='green',
                line=dict(width=2, color='darkgreen')
            ),
            name='Buy Signal'
        ))
        
        # Add sell signals
        sell_signals = signals_df[signals_df['Signal'] == 'Sell']
        fig_signals.add_trace(go.Scatter(
            x=sell_signals.index,
            y=df.loc[sell_signals.index, 'Close'],
            mode='markers',
            marker=dict(
                size=10,
                symbol='triangle-down',
                color='red',
                line=dict(width=2, color='darkred')
            ),
            name='Sell Signal'
        ))
        
        fig_signals.update_layout(
            title='Trading Signals',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig_signals, use_container_width=True)
        
        # Signal history table
        st.subheader("Signal History")
        
        signal_history = signals_df[['Signal', 'Confidence', 'Predicted_Move', 'Actual_Move']].iloc[::-1]
        signal_history = signal_history.reset_index()
        signal_history.columns = ['Date', 'Signal', 'Confidence (%)', 'Predicted Move (%)', 'Actual Move (%)']
        signal_history['Date'] = signal_history['Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(signal_history, use_container_width=True)
        
    else:
        st.error(f"No data found for symbol {stock_symbol}. Please check the symbol and try again.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
