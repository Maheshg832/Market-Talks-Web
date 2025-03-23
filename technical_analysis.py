import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.data_fetcher import get_stock_data, get_stock_info
from utils.technical_indicators import calculate_indicators, identify_patterns

st.set_page_config(
    page_title="Technical Analysis - Stock Analysis Platform",
    page_icon="📈",
    layout="wide"
)

# Session state initialization
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = True

# Sidebar
with st.sidebar:
    st.title("Technical Analysis")
    
    # Dark mode toggle
    st.session_state['dark_mode'] = st.toggle("Dark Mode", value=st.session_state['dark_mode'])
    
    # Stock search
    stock_symbol = st.text_input("Stock Symbol", "AAPL").upper()
    
    # Date range selection
    today = datetime.now()
    start_date = st.date_input("Start Date", today - timedelta(days=365))
    end_date = st.date_input("End Date", today)
    
    # Chart type selection
    chart_type = st.selectbox(
        "Chart Type",
        ["Candlestick", "OHLC", "Line"]
    )
    
    # Indicators selection
    st.subheader("Technical Indicators")
    
    show_sma = st.checkbox("Simple Moving Averages", True)
    show_ema = st.checkbox("Exponential Moving Averages", False)
    show_bollinger = st.checkbox("Bollinger Bands", True)
    show_rsi = st.checkbox("RSI", True)
    show_macd = st.checkbox("MACD", True)
    show_volume = st.checkbox("Volume", True)
    show_patterns = st.checkbox("Pattern Recognition", False)

# Main content
st.title("Technical Analysis")

try:
    # Fetch stock data
    df = get_stock_data(stock_symbol, start_date, end_date)
    
    if df is not None and not df.empty:
        # Get stock info
        stock_info = get_stock_info(stock_symbol)
        
        if stock_info:
            # Display stock info
            st.subheader(f"{stock_info['name']} ({stock_symbol})")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sector", stock_info['sector'])
            col2.metric("Industry", stock_info['industry'])
            
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
                col3.metric("Market Cap", formatted_cap)
            else:
                col3.metric("Market Cap", "N/A")
            
            # Format P/E ratio
            pe_ratio = stock_info['pe_ratio']
            if isinstance(pe_ratio, (int, float)):
                col4.metric("P/E Ratio", f"{pe_ratio:.2f}")
            else:
                col4.metric("P/E Ratio", "N/A")
        
        # Calculate indicators
        indicators = calculate_indicators(df)
        
        # Identify patterns if selected
        patterns = {}
        if show_patterns:
            patterns = identify_patterns(df)
        
        # Create figure layout based on selected indicators
        fig_rows = 2  # Main chart + at least one indicator
        if show_rsi and show_macd:
            fig_rows += 1
        if show_volume:
            fig_rows += 1
        
        # Create subplots
        fig = make_subplots(
            rows=fig_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5] + [0.5/(fig_rows-1)] * (fig_rows-1)
        )
        
        # Add main price chart
        if chart_type == "Candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="OHLC"
                ),
                row=1, col=1
            )
        elif chart_type == "OHLC":
            fig.add_trace(
                go.Ohlc(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="OHLC"
                ),
                row=1, col=1
            )
        else:  # Line chart
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name="Close"
                ),
                row=1, col=1
            )
        
        # Add moving averages if selected
        if show_sma:
            for period in [20, 50, 200]:
                sma = df['Close'].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=sma,
                        mode='lines',
                        name=f"SMA {period}",
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
        
        if show_ema:
            for period in [12, 26, 50]:
                ema = df['Close'].ewm(span=period, adjust=False).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=ema,
                        mode='lines',
                        name=f"EMA {period}",
                        line=dict(width=1, dash='dash')
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands if selected
        if show_bollinger:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=indicators['BB_Upper'],
                    mode='lines',
                    name="BB Upper",
                    line=dict(width=1, dash='dash')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=indicators['BB_Middle'],
                    mode='lines',
                    name="BB Middle",
                    line=dict(width=1, dash='dot')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=indicators['BB_Lower'],
                    mode='lines',
                    name="BB Lower",
                    line=dict(width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # Add patterns if selected
        if show_patterns and patterns:
            for pattern, date in patterns.items():
                if pattern in ['support', 'resistance']:
                    fig.add_hline(
                        y=patterns[pattern],
                        line_dash="dash",
                        line_color="white",
                        annotation_text=pattern.capitalize(),
                        row=1, col=1
                    )
                elif isinstance(date, pd.Timestamp):
                    # Add marker for pattern
                    idx = df.index.get_loc(date)
                    price = df['Close'].iloc[idx]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[date],
                            y=[price],
                            mode='markers',
                            marker=dict(
                                size=12,
                                symbol='x',
                                color='yellow'
                            ),
                            name=pattern.replace('_', ' ').capitalize(),
                            hoverinfo='text',
                            hovertext=f"{pattern.replace('_', ' ').capitalize()} pattern"
                        ),
                        row=1, col=1
                    )
        
        # Current row for indicators
        current_row = 2
        
        # Add RSI if selected
        if show_rsi:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=indicators['RSI'],
                    mode='lines',
                    name="RSI"
                ),
                row=current_row, col=1
            )
            
            # Add oversold/overbought lines
            fig.add_hline(
                y=70, 
                line_dash="dash", 
                line_color="red",
                row=current_row, col=1
            )
            
            fig.add_hline(
                y=30, 
                line_dash="dash", 
                line_color="green",
                row=current_row, col=1
            )
            
            # Add annotation
            fig.update_yaxes(title_text="RSI", row=current_row, col=1)
            
            # Increment row counter if we also show MACD
            if show_macd:
                current_row += 1
        
        # Add MACD if selected
        if show_macd:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=indicators['MACD'],
                    mode='lines',
                    name="MACD"
                ),
                row=current_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=indicators['MACD_Signal'],
                    mode='lines',
                    name="Signal"
                ),
                row=current_row, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=indicators['MACD_Hist'],
                    name="Histogram"
                ),
                row=current_row, col=1
            )
            
            # Add annotation
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
            
            # Increment row counter if we also show volume
            if show_volume:
                current_row += 1
        
        # Add volume if selected
        if show_volume:
            # Color volume bars based on price change
            colors = ['green' if close > open else 'red' for open, close in zip(df['Open'], df['Close'])]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name="Volume",
                    marker=dict(
                        color=colors
                    )
                ),
                row=current_row, col=1
            )
            
            # Add annotation
            fig.update_yaxes(title_text="Volume", row=current_row, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{stock_symbol} Technical Analysis",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
            xaxis_rangeslider_visible=False,
            height=800,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Display additional technical analysis
        st.subheader("Indicator Values (Latest)")
        
        # Get the latest indicator values
        latest_indicators = indicators.iloc[-1].to_dict()
        
        # Create columns for display
        col1, col2, col3, col4 = st.columns(4)
        
        # RSI interpretation
        rsi_value = latest_indicators['RSI']
        if pd.notna(rsi_value):
            rsi_text = f"{rsi_value:.2f}"
            if rsi_value > 70:
                rsi_status = "Overbought"
            elif rsi_value < 30:
                rsi_status = "Oversold"
            else:
                rsi_status = "Neutral"
            
            col1.metric("RSI", rsi_text, rsi_status)
        else:
            col1.metric("RSI", "N/A")
        
        # MACD interpretation
        macd_value = latest_indicators['MACD']
        signal_value = latest_indicators['MACD_Signal']
        
        if pd.notna(macd_value) and pd.notna(signal_value):
            macd_diff = macd_value - signal_value
            macd_text = f"{macd_value:.2f}"
            
            if macd_diff > 0:
                macd_status = "Bullish"
            else:
                macd_status = "Bearish"
            
            col2.metric("MACD", macd_text, macd_status)
        else:
            col2.metric("MACD", "N/A")
        
        # Bollinger Bands
        close_price = df['Close'].iloc[-1]
        bb_upper = latest_indicators['BB_Upper']
        bb_lower = latest_indicators['BB_Lower']
        
        if pd.notna(bb_upper) and pd.notna(bb_lower):
            bb_percent = (close_price - bb_lower) / (bb_upper - bb_lower) * 100
            bb_text = f"{bb_percent:.2f}%"
            
            if bb_percent > 80:
                bb_status = "Near Upper Band"
            elif bb_percent < 20:
                bb_status = "Near Lower Band"
            else:
                bb_status = "Middle Range"
            
            col3.metric("Bollinger %B", bb_text, bb_status)
        else:
            col3.metric("Bollinger %B", "N/A")
        
        # SMA comparison
        sma50 = df['Close'].rolling(window=50).mean().iloc[-1]
        sma200 = df['Close'].rolling(window=200).mean().iloc[-1]
        
        if pd.notna(sma50) and pd.notna(sma200):
            sma_diff = ((sma50 / sma200) - 1) * 100
            sma_text = f"{sma_diff:.2f}%"
            
            if sma_diff > 0:
                sma_status = "Golden Cross"
            else:
                sma_status = "Death Cross"
            
            col4.metric("SMA50 vs SMA200", sma_text, sma_status)
        else:
            col4.metric("SMA50 vs SMA200", "N/A")
        
        # Pattern analysis
        if show_patterns and patterns:
            st.subheader("Pattern Analysis")
            
            # Display identified patterns
            pattern_text = ""
            for pattern, value in patterns.items():
                if pattern == 'support':
                    pattern_text += f"• Support level identified at ${value:.2f}\n"
                elif pattern == 'resistance':
                    pattern_text += f"• Resistance level identified at ${value:.2f}\n"
                elif isinstance(value, pd.Timestamp):
                    pattern_text += f"• {pattern.replace('_', ' ').capitalize()} pattern identified on {value.strftime('%Y-%m-%d')}\n"
            
            if pattern_text:
                st.markdown(pattern_text)
            else:
                st.info("No significant patterns identified in the current time frame.")
        
        # Technical summary
        st.subheader("Technical Summary")
        
        # Generate summary based on indicators
        summary_points = []
        
        # RSI Summary
        if pd.notna(rsi_value):
            if rsi_value > 70:
                summary_points.append("• RSI indicates **overbought** conditions, suggesting potential downward pressure.")
            elif rsi_value < 30:
                summary_points.append("• RSI indicates **oversold** conditions, suggesting potential upward pressure.")
            else:
                summary_points.append(f"• RSI at {rsi_value:.2f} indicates **neutral** momentum.")
        
        # MACD Summary
        if pd.notna(macd_value) and pd.notna(signal_value):
            if macd_value > signal_value:
                summary_points.append("• MACD above signal line suggests **bullish** momentum.")
            else:
                summary_points.append("• MACD below signal line suggests **bearish** momentum.")
        
        # Moving Average Summary
        if pd.notna(sma50) and pd.notna(sma200):
            if sma50 > sma200:
                summary_points.append("• Price trading above both 50-day and 200-day moving averages, indicating **bullish** trend.")
            elif close_price > sma50 > sma200:
                summary_points.append("• Price above 50-day MA, which is above 200-day MA, indicating **strong bullish** trend.")
            elif sma50 < sma200 and close_price < sma50:
                summary_points.append("• Price below 50-day MA, which is below 200-day MA, indicating **strong bearish** trend.")
            else:
                summary_points.append("• Mixed signals from moving averages, indicating **consolidation** or **trend change**.")
        
        # Bollinger Bands Summary
        if pd.notna(bb_upper) and pd.notna(bb_lower):
            if close_price > bb_upper:
                summary_points.append("• Price above upper Bollinger Band, indicating **strong bullish** momentum but potentially **overbought**.")
            elif close_price < bb_lower:
                summary_points.append("• Price below lower Bollinger Band, indicating **strong bearish** momentum but potentially **oversold**.")
            else:
                band_width = (bb_upper - bb_lower) / latest_indicators['BB_Middle']
                if band_width < 0.1:  # Narrow bands
                    summary_points.append("• Narrow Bollinger Bands indicate **low volatility**, potential for **breakout** ahead.")
                else:
                    summary_points.append("• Price within Bollinger Bands indicates **normal trading conditions**.")
        
        # Display summary
        if summary_points:
            for point in summary_points:
                st.markdown(point)
        else:
            st.info("Insufficient data for technical summary.")
    else:
        st.error(f"No data found for symbol {stock_symbol}. Please check the symbol and try again.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Stock Market Analysis Platform | Technical Analysis")
