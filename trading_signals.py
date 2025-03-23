import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_fetcher import get_stock_data
from utils.ml_models import generate_signals, generate_features
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Trading Signals - Stock Analysis Platform",
    page_icon="🤖",
    layout="wide"
)

# Session state initialization
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = True

# Sidebar
with st.sidebar:
    st.title("AI Trading Signals")
    
    # Dark mode toggle
    st.session_state['dark_mode'] = st.toggle("Dark Mode", value=st.session_state['dark_mode'])
    
    # Stock search
    stock_symbol = st.text_input("Stock Symbol", "AAPL").upper()
    
    # Date range selection
    today = datetime.now()
    start_date = st.date_input("Start Date", today - timedelta(days=730))  # 2 years data
    end_date = st.date_input("End Date", today)
    
    # AI model parameters
    st.subheader("Model Parameters")
    
    confidence_threshold = st.slider(
        "Signal Confidence Threshold (%)",
        min_value=50,
        max_value=90,
        value=60,
        step=5
    )
    
    training_window = st.slider(
        "Training Window (Days)",
        min_value=30,
        max_value=252,
        value=60,
        step=30
    )
    
    prediction_horizon = st.slider(
        "Prediction Horizon (Days)",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

# Main content
st.title("AI Trading Signals")

try:
    # Fetch stock data
    df = get_stock_data(stock_symbol, start_date, end_date)
    
    if df is not None and not df.empty:
        # Display current price information
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        price_change = current_price - prev_close
        pct_change = (price_change / prev_close) * 100
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            label=f"{stock_symbol} Price", 
            value=f"${current_price:.2f}", 
            delta=f"{price_change:.2f} ({pct_change:.2f}%)"
        )
        
        # Generate signals
        signals_df = generate_signals(df)
        
        if not signals_df.empty:
            # Latest signal
            latest_signal = signals_df.iloc[-1]
            signal_type = latest_signal['Signal']
            confidence = latest_signal['Confidence']
            predicted_move = latest_signal['Predicted_Move']
            
            signal_color = "green" if signal_type == "Buy" else "red"
            col2.metric(
                label="Current Signal", 
                value=signal_type,
                delta=f"{confidence:.1f}% Confidence"
            )
            
            col3.metric(
                label="Predicted Move", 
                value=f"{predicted_move:.2f}%",
                delta=f"{prediction_horizon} Days"
            )
            
            # Signal history chart
            st.subheader("Price and Signals History")
            
            # Create figure
            fig = go.Figure()
            
            # Add close price
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Close Price'
            ))
            
            # Add buy signals
            buy_signals = signals_df[signals_df['Signal'] == 'Buy']
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
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
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
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
            
            fig.update_layout(
                title='Trading Signals',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
                height=500,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model performance
            st.subheader("Model Performance")
            
            # Calculate accuracy metrics
            accuracy = (signals_df['Actual_Move'] * signals_df['Predicted_Move'] > 0).mean() * 100
            
            # Calculate average return when following signals
            buy_returns = signals_df.loc[signals_df['Signal'] == 'Buy', 'Actual_Move'].mean()
            sell_returns = -signals_df.loc[signals_df['Signal'] == 'Sell', 'Actual_Move'].mean()
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            col1.metric(
                label="Signal Direction Accuracy", 
                value=f"{accuracy:.1f}%"
            )
            
            col2.metric(
                label="Avg Return on Buy Signals", 
                value=f"{buy_returns:.2f}%"
            )
            
            col3.metric(
                label="Avg Return on Sell Signals", 
                value=f"{sell_returns:.2f}%"
            )
            
            # Signal history
            st.subheader("Signal History")
            
            # Prepare signal history for display
            signal_history = signals_df.copy()
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
            
            # Backtesting simulation
            st.subheader("Trading Strategy Backtest")
            
            # Create backtest DataFrame
            backtest = pd.DataFrame(index=df.index)
            backtest['Close'] = df['Close']
            
            # Add signals
            backtest['Signal'] = None
            for idx, row in signals_df.iterrows():
                backtest.loc[idx, 'Signal'] = row['Signal']
            
            # Forward fill signals
            backtest['Signal'] = backtest['Signal'].ffill()
            
            # Calculate returns
            backtest['Daily_Return'] = backtest['Close'].pct_change()
            
            # Calculate strategy returns
            backtest['Strategy_Return'] = 0.0
            backtest.loc[backtest['Signal'] == 'Buy', 'Strategy_Return'] = backtest['Daily_Return']
            backtest.loc[backtest['Signal'] == 'Sell', 'Strategy_Return'] = -backtest['Daily_Return']
            
            # Calculate cumulative returns
            backtest['Cum_Market_Return'] = (1 + backtest['Daily_Return']).cumprod() - 1
            backtest['Cum_Strategy_Return'] = (1 + backtest['Strategy_Return']).cumprod() - 1
            
            # Create backtest chart
            fig_backtest = go.Figure()
            
            # Add market returns
            fig_backtest.add_trace(go.Scatter(
                x=backtest.index,
                y=backtest['Cum_Market_Return'] * 100,
                mode='lines',
                name='Buy and Hold'
            ))
            
            # Add strategy returns
            fig_backtest.add_trace(go.Scatter(
                x=backtest.index,
                y=backtest['Cum_Strategy_Return'] * 100,
                mode='lines',
                name='Signal Strategy'
            ))
            
            fig_backtest.update_layout(
                title='Backtest Performance',
                xaxis_title='Date',
                yaxis_title='Cumulative Return (%)',
                template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
                height=400,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_backtest, use_container_width=True)
            
            # Backtest statistics
            st.subheader("Backtest Statistics")
            
            # Calculate statistics
            market_return = backtest['Cum_Market_Return'].iloc[-1] * 100
            strategy_return = backtest['Cum_Strategy_Return'].iloc[-1] * 100
            
            # Calculate annualized return
            days = (backtest.index[-1] - backtest.index[0]).days
            market_annual = ((1 + market_return/100) ** (365/days) - 1) * 100
            strategy_annual = ((1 + strategy_return/100) ** (365/days) - 1) * 100
            
            # Calculate sharpe ratio (assuming 0% risk-free rate)
            market_daily_returns = backtest['Daily_Return'].fillna(0)
            strategy_daily_returns = backtest['Strategy_Return'].fillna(0)
            
            market_sharpe = np.sqrt(252) * market_daily_returns.mean() / market_daily_returns.std()
            strategy_sharpe = np.sqrt(252) * strategy_daily_returns.mean() / strategy_daily_returns.std()
            
            # Calculate max drawdown
            market_cum = (1 + market_daily_returns).cumprod()
            strategy_cum = (1 + strategy_daily_returns).cumprod()
            
            market_drawdown = (market_cum / market_cum.cummax() - 1).min() * 100
            strategy_drawdown = (strategy_cum / strategy_cum.cummax() - 1).min() * 100
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                label="Total Return", 
                value=f"{strategy_return:.2f}%",
                delta=f"{strategy_return - market_return:.2f}%"
            )
            
            col2.metric(
                label="Annualized Return", 
                value=f"{strategy_annual:.2f}%",
                delta=f"{strategy_annual - market_annual:.2f}%"
            )
            
            col3.metric(
                label="Sharpe Ratio", 
                value=f"{strategy_sharpe:.2f}",
                delta=f"{strategy_sharpe - market_sharpe:.2f}"
            )
            
            col4.metric(
                label="Max Drawdown", 
                value=f"{strategy_drawdown:.2f}%",
                delta=f"{strategy_drawdown - market_drawdown:.2f}%"
            )
            
            # Model explanation
            st.subheader("Model Feature Importance")
            
            # Generate features
            features = generate_features(df)
            
            if not features.empty:
                # Train a model for feature importance visualization
                X = features.drop(columns=['Close', 'Volume'])
                y = features['Close'].pct_change(prediction_horizon).shift(-prediction_horizon)
                y = (y > 0).astype(int)  # Binary classification
                
                # Filter out NaN values
                mask = ~np.isnan(y)
                X = X[mask]
                y = y[mask]
                
                if len(X) > 0 and len(y) > 0:
                    # Train a simple model
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    
                    # Get feature importance
                    importance = model.feature_importances_
                    
                    # Create feature importance DataFrame
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': importance
                    })
                    
                    # Sort by importance
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    # Display top 10 features
                    fig_importance = go.Figure()
                    
                    fig_importance.add_trace(go.Bar(
                        x=importance_df['Importance'][:10],
                        y=importance_df['Feature'][:10],
                        orientation='h'
                    ))
                    
                    fig_importance.update_layout(
                        title='Top 10 Feature Importance',
                        xaxis_title='Importance',
                        yaxis_title='Feature',
                        template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
                        height=400,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Brief explanation of the model
                    st.subheader("How the AI Trading Model Works")
                    
                    st.write("""
                    The AI trading model uses machine learning algorithms to identify patterns in stock price movements and generate trading signals. Here's how it works:
                    
                    1. **Feature Generation**: The model calculates various technical indicators and price patterns from historical data.
                    
                    2. **Training**: The model is trained on historical data to learn the relationship between these features and future price movements.
                    
                    3. **Signal Generation**: Based on the learned patterns, the model predicts future price movements and generates buy or sell signals.
                    
                    4. **Confidence Calculation**: The model assigns a confidence score to each signal based on the probability of the predicted outcome.
                    
                    The feature importance chart above shows which indicators and patterns have the most influence on the model's predictions.
                    """)
                    
                    # Disclaimer
                    st.warning("""
                    **Disclaimer**: AI trading signals are for informational purposes only and should not be considered as financial advice. 
                    Past performance is not indicative of future results. Always conduct your own research before making investment decisions.
                    """)
                else:
                    st.info("Insufficient data for feature importance analysis.")
            else:
                st.info("Insufficient data for feature importance analysis.")
        else:
            st.info("No trading signals available for the selected time period. Try extending the date range.")
    else:
        st.error(f"No data found for symbol {stock_symbol}. Please check the symbol and try again.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Stock Market Analysis Platform | AI Trading Signals")
