import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_fetcher import get_stock_data, get_multiple_stocks_data, get_stock_info
from utils.technical_indicators import calculate_indicators
from utils.ml_models import generate_signals

st.set_page_config(
    page_title="Watchlist - Stock Analysis Platform",
    page_icon="📋",
    layout="wide"
)

# Session state initialization
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = True

if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Sidebar
with st.sidebar:
    st.title("Watchlist")
    
    # Dark mode toggle
    st.session_state['dark_mode'] = st.toggle("Dark Mode", value=st.session_state['dark_mode'])
    
    # Stock search and add to watchlist
    new_stock = st.text_input("Add Stock to Watchlist").upper()
    
    if st.button("Add Stock") and new_stock:
        if new_stock not in st.session_state['watchlist']:
            st.session_state['watchlist'].append(new_stock)
            st.success(f"Added {new_stock} to watchlist!")
            st.rerun()
    
    # Date range selection
    today = datetime.now()
    start_date = st.date_input("Start Date", today - timedelta(days=180))
    end_date = st.date_input("End Date", today)
    
    # Comparison parameter
    compare_method = st.selectbox(
        "Comparison Method",
        ["Percent Change", "Normalized (Base 100)", "Absolute Price"]
    )

# Main content
st.title("Watchlist")

if not st.session_state['watchlist']:
    st.info("Your watchlist is empty. Add stocks using the sidebar.")
else:
    # Fetch data for all stocks in watchlist
    watchlist_data = get_multiple_stocks_data(st.session_state['watchlist'], start_date, end_date)
    
    # Handle any invalid symbols
    valid_symbols = [symbol for symbol, data in watchlist_data.items() if data is not None and not data.empty]
    invalid_symbols = [symbol for symbol in st.session_state['watchlist'] if symbol not in valid_symbols]
    
    if invalid_symbols:
        st.warning(f"Could not fetch data for these symbols: {', '.join(invalid_symbols)}")
        # Remove invalid symbols from watchlist
        st.session_state['watchlist'] = valid_symbols
    
    if valid_symbols:
        # Create watchlist table
        watchlist_data_rows = []
        
        for symbol, data in watchlist_data.items():
            if data is not None and not data.empty:
                # Get latest data
                latest = data.iloc[-1]
                prev_day = data.iloc[-2]
                
                # Calculate daily change
                daily_change = (latest['Close'] - prev_day['Close']) / prev_day['Close'] * 100
                
                # Calculate 1-month change
                month_ago_idx = max(0, len(data) - 22)
                month_ago = data.iloc[month_ago_idx]
                monthly_change = (latest['Close'] - month_ago['Close']) / month_ago['Close'] * 100
                
                # Calculate indicators
                indicators = calculate_indicators(data)
                latest_indicators = indicators.iloc[-1]
                
                # Generate trading signal
                signals = generate_signals(data)
                latest_signal = "N/A"
                signal_confidence = 0
                
                if not signals.empty:
                    latest_signal = signals.iloc[-1]['Signal']
                    signal_confidence = signals.iloc[-1]['Confidence']
                
                # Add to data rows
                watchlist_data_rows.append({
                    'Symbol': symbol,
                    'Last Price': latest['Close'],
                    'Daily Change (%)': daily_change,
                    'Monthly Change (%)': monthly_change,
                    'Volume': latest['Volume'],
                    'RSI': latest_indicators['RSI'],
                    'Signal': latest_signal,
                    'Confidence (%)': signal_confidence
                })
        
        # Create DataFrame
        watchlist_df = pd.DataFrame(watchlist_data_rows)
        
        # Apply formatting
        watchlist_df['Last Price'] = watchlist_df['Last Price'].map('${:,.2f}'.format)
        watchlist_df['Daily Change (%)'] = watchlist_df['Daily Change (%)'].map('{:+.2f}%'.format)
        watchlist_df['Monthly Change (%)'] = watchlist_df['Monthly Change (%)'].map('{:+.2f}%'.format)
        watchlist_df['Volume'] = watchlist_df['Volume'].map('{:,}'.format)
        watchlist_df['RSI'] = watchlist_df['RSI'].map('{:.1f}'.format)
        watchlist_df['Confidence (%)'] = watchlist_df['Confidence (%)'].map('{:.1f}'.format)
        
        # Display watchlist table
        st.dataframe(watchlist_df, use_container_width=True)
        
        # Watchlist management section
        st.subheader("Manage Watchlist")
        
        # Create multi-select for stocks
        selected_to_remove = st.multiselect(
            "Select stocks to remove",
            options=valid_symbols
        )
        
        # Remove button
        if st.button("Remove Selected") and selected_to_remove:
            for symbol in selected_to_remove:
                if symbol in st.session_state['watchlist']:
                    st.session_state['watchlist'].remove(symbol)
            st.success(f"Removed {len(selected_to_remove)} stocks from watchlist")
            st.rerun()
        
        # Performance comparison chart
        st.subheader("Performance Comparison")
        
        fig = go.Figure()
        
        for symbol, data in watchlist_data.items():
            if data is not None and not data.empty:
                if compare_method == "Percent Change":
                    # Calculate percent change from first day
                    first_close = data['Close'].iloc[0]
                    pct_change = ((data['Close'] / first_close) - 1) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=pct_change,
                        mode='lines',
                        name=symbol
                    ))
                    
                    y_axis_title = "Percent Change (%)"
                
                elif compare_method == "Normalized (Base 100)":
                    # Normalize to 100
                    first_close = data['Close'].iloc[0]
                    normalized = (data['Close'] / first_close) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=normalized,
                        mode='lines',
                        name=symbol
                    ))
                    
                    y_axis_title = "Normalized Price (Base 100)"
                
                else:  # Absolute Price
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name=symbol
                    ))
                    
                    y_axis_title = "Price (USD)"
        
        fig.update_layout(
            title='Watchlist Performance Comparison',
            xaxis_title='Date',
            yaxis_title=y_axis_title,
            template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
            height=500,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual stock analysis
        st.subheader("Individual Stock Analysis")
        
        # Select a stock for detailed analysis
        selected_stock = st.selectbox("Select a stock for detailed analysis", valid_symbols)
        
        if selected_stock:
            # Get the data for the selected stock
            stock_data = watchlist_data[selected_stock]
            
            if stock_data is not None and not stock_data.empty:
                # Current price information
                current_price = stock_data['Close'].iloc[-1]
                prev_close = stock_data['Close'].iloc[-2]
                price_change = current_price - prev_close
                pct_change = (price_change / prev_close) * 100
                
                # Get stock info
                stock_info = get_stock_info(selected_stock)
                
                if stock_info:
                    # Display stock info
                    st.subheader(f"{stock_info['name']} ({selected_stock})")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric(
                        label="Current Price", 
                        value=f"${current_price:.2f}", 
                        delta=f"{price_change:.2f} ({pct_change:.2f}%)"
                    )
                    
                    col2.metric("Sector", stock_info['sector'])
                    col3.metric("Industry", stock_info['industry'])
                    
                    # Format market cap
                    market_cap = stock_info['market_cap']
                    if isinstance(market_cap, (int, float)):
                        if market_cap >= 1e12:
                            formatted_cap = f"${market_cap/1e12:.2f}T"
                        elif market_cap >= 1e9:
                            formatted_cap = f"${market_cap/1e9:.2f}B"
                        elif market_cap >= 1e6:
                            formatted_cap = f"${market_cap/1e6:.2f}M"
                        else:
                            formatted_cap = f"${market_cap:,.0f}"
                        col4.metric("Market Cap", formatted_cap)
                    else:
                        col4.metric("Market Cap", "N/A")
                    
                    # Display business summary
                    if 'business_summary' in stock_info and stock_info['business_summary'] != 'N/A':
                        with st.expander("Business Summary"):
                            st.write(stock_info['business_summary'])
                
                # Calculate indicators
                indicators = calculate_indicators(stock_data)
                
                # Create stock chart
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name="OHLC"
                ))
                
                # Add volume as bar chart at the bottom
                fig.add_trace(go.Bar(
                    x=stock_data.index,
                    y=stock_data['Volume'],
                    name="Volume",
                    marker=dict(
                        color='rgba(128, 128, 128, 0.5)'
                    ),
                    yaxis="y2"
                ))
                
                # Add some moving averages
                for period in [20, 50, 200]:
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'].rolling(window=period).mean(),
                        mode='lines',
                        name=f"SMA {period}",
                        line=dict(width=1)
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"{selected_stock} Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right",
                        showgrid=False
                    ),
                    template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
                    height=500,
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical indicators section
                st.subheader("Technical Indicators")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI chart
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
                        title='Relative Strength Index (RSI)',
                        xaxis_title='Date',
                        yaxis_title='RSI',
                        template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
                        height=300,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    # MACD chart
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
                        title='Moving Average Convergence Divergence (MACD)',
                        xaxis_title='Date',
                        yaxis_title='MACD',
                        template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
                        height=300,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                # Bollinger Bands chart
                fig_bb = go.Figure()
                
                fig_bb.add_trace(go.Scatter(
                    x=indicators.index,
                    y=stock_data['Close'],
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
                    title='Bollinger Bands',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig_bb, use_container_width=True)
                
                # Trading signals
                st.subheader("Trading Signals")
                
                signals = generate_signals(stock_data)
                
                if not signals.empty:
                    # Create signals history table
                    signal_history = signals.copy()
                    signal_history = signal_history.reset_index()
                    signal_history['Date'] = signal_history['Date'].dt.strftime('%Y-%m-%d')
                    signal_history = signal_history[['Date', 'Signal', 'Confidence', 'Predicted_Move', 'Actual_Move']]
                    signal_history.columns = ['Date', 'Signal', 'Confidence (%)', 'Predicted Move (%)', 'Actual Move (%)']
                    
                    # Format numeric columns
                    signal_history['Confidence (%)'] = signal_history['Confidence (%)'].apply(lambda x: f"{x:.1f}")
                    signal_history['Predicted Move (%)'] = signal_history['Predicted Move (%)'].apply(lambda x: f"{x:.2f}")
                    signal_history['Actual Move (%)'] = signal_history['Actual Move (%)'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "Pending")
                    
                    # Sort by date (descending)
                    signal_history = signal_history.sort_values('Date', ascending=False)
                    
                    # Display table
                    st.dataframe(signal_history, use_container_width=True)
                    
                    # Add a note about the signals
                    st.info("""
                    The signals shown above are based on a machine learning model that analyzes historical price patterns and technical indicators.
                    The model predicts the likelihood of price movements over the next few days and generates buy or sell signals accordingly.
                    """)
                else:
                    st.info("No trading signals available for the selected stock in the current time period.")
            else:
                st.error(f"No data found for {selected_stock}.")

# Footer
st.markdown("---")
st.markdown("Stock Market Analysis Platform | Watchlist")
