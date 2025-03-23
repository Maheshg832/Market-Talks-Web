import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils.news_fetcher import get_market_news, get_stock_news
from utils.data_fetcher import get_stock_data, get_multiple_stocks_data

st.set_page_config(
    page_title="Market News - Stock Analysis Platform",
    page_icon="📰",
    layout="wide"
)

# Session state initialization
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = True

if 'selected_news_tab' not in st.session_state:
    st.session_state['selected_news_tab'] = "General Market News"

# Sidebar
with st.sidebar:
    st.title("Market News")
    
    # Dark mode toggle
    st.session_state['dark_mode'] = st.toggle("Dark Mode", value=st.session_state['dark_mode'])
    
    # News category selection
    st.session_state['selected_news_tab'] = st.radio(
        "News Category",
        ["General Market News", "Stock-Specific News", "Economic Calendar"]
    )
    
    # Stock selection for stock-specific news
    if st.session_state['selected_news_tab'] == "Stock-Specific News":
        stock_symbol = st.text_input("Stock Symbol", "AAPL").upper()
    
    # Date range for economic calendar
    if st.session_state['selected_news_tab'] == "Economic Calendar":
        today = datetime.now()
        start_date = st.date_input("Start Date", today - timedelta(days=7))
        end_date = st.date_input("End Date", today + timedelta(days=7))

# Main content
st.title("Market News and Events")

# News content based on selection
if st.session_state['selected_news_tab'] == "General Market News":
    st.header("General Market News")
    
    # Fetch market news
    news = get_market_news(limit=15)
    
    if news:
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Main news section
            st.subheader("Top Market News")
            
            # Display top news in cards
            for i, article in enumerate(news[:5]):
                with st.container():
                    st.markdown(f"### {article['title']}")
                    st.markdown(f"**Source:** {article['source']['name']} | **Published:** {article['publishedAt'][:10]}")
                    st.markdown(article['description'])
                    st.markdown(f"[Read more]({article['url']})")
                    st.markdown("---")
        
        with col2:
            # Market summary section
            st.subheader("Market Summary")
            
            # Fetch data for major indices
            indices = ['^GSPC', '^DJI', '^IXIC']
            indices_names = {
                '^GSPC': 'S&P 500',
                '^DJI': 'Dow Jones',
                '^IXIC': 'NASDAQ'
            }
            
            # Get current date and date 1 day ago
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2)  # Get 2 days to ensure we have at least 1 full trading day
            
            try:
                # Fetch data
                indices_data = get_multiple_stocks_data(indices, start_date, end_date)
                
                if indices_data:
                    # Create metrics
                    for symbol, data in indices_data.items():
                        if data is not None and not data.empty:
                            current_price = data['Close'].iloc[-1]
                            prev_close = data['Close'].iloc[-2]
                            price_change = current_price - prev_close
                            pct_change = (price_change / prev_close) * 100
                            
                            st.metric(
                                label=indices_names.get(symbol, symbol), 
                                value=f"{current_price:.2f}", 
                                delta=f"{pct_change:.2f}%"
                            )
            except Exception as e:
                st.error(f"Error fetching market data: {str(e)}")
            
            # Recent news headlines
            st.subheader("Recent Headlines")
            
            for article in news[5:15]:
                st.markdown(f"• [{article['title']}]({article['url']})")
    else:
        st.info("No market news available at the moment.")

elif st.session_state['selected_news_tab'] == "Stock-Specific News":
    st.header(f"News for {stock_symbol}")
    
    # Fetch stock data for display
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        # Fetch data
        df = get_stock_data(stock_symbol, start_date, end_date)
        
        if df is not None and not df.empty:
            # Display price chart
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            ))
            
            fig.update_layout(
                title=f"{stock_symbol} - Last 30 Days",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template='plotly_dark' if st.session_state['dark_mode'] else 'plotly_white',
                xaxis_rangeslider_visible=False,
                height=300,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
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
            
            # Fetch news specific to the stock
            news = get_stock_news(stock_symbol, limit=10)
            
            if news:
                st.subheader(f"Latest News for {stock_symbol}")
                
                # Display news in expandable format
                for article in news:
                    with st.expander(article['title']):
                        st.write(f"**Source:** {article['source']['name']}")
                        st.write(f"**Published:** {article['publishedAt'][:10]}")
                        st.write(article['description'])
                        st.write(f"[Read more]({article['url']})")
            else:
                st.info(f"No recent news found for {stock_symbol}.")
        else:
            st.error(f"No data found for symbol {stock_symbol}. Please check the symbol and try again.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

elif st.session_state['selected_news_tab'] == "Economic Calendar":
    st.header("Economic Calendar")
    
    # Mock economic calendar data (since we don't have a real API for this)
    # In a real application, you would integrate with an economic calendar API
    
    # Create mock data
    calendar_data = [
        {
            "date": "2023-11-01",
            "time": "08:30",
            "event": "ADP Employment Change",
            "country": "US",
            "importance": "High",
            "actual": "113K",
            "forecast": "150K",
            "previous": "89K"
        },
        {
            "date": "2023-11-01",
            "time": "14:00",
            "event": "Fed Interest Rate Decision",
            "country": "US",
            "importance": "High",
            "actual": "5.50%",
            "forecast": "5.50%",
            "previous": "5.50%"
        },
        {
            "date": "2023-11-02",
            "time": "08:30",
            "event": "Initial Jobless Claims",
            "country": "US",
            "importance": "Medium",
            "actual": "217K",
            "forecast": "210K",
            "previous": "212K"
        },
        {
            "date": "2023-11-03",
            "time": "08:30",
            "event": "Nonfarm Payrolls",
            "country": "US",
            "importance": "High",
            "actual": "150K",
            "forecast": "180K",
            "previous": "336K"
        },
        {
            "date": "2023-11-03",
            "time": "08:30",
            "event": "Unemployment Rate",
            "country": "US",
            "importance": "High",
            "actual": "3.9%",
            "forecast": "3.8%",
            "previous": "3.8%"
        },
        {
            "date": "2023-11-07",
            "time": "10:00",
            "event": "JOLTs Job Openings",
            "country": "US",
            "importance": "Medium",
            "actual": None,
            "forecast": "9.25M",
            "previous": "9.61M"
        },
        {
            "date": "2023-11-09",
            "time": "08:30",
            "event": "Initial Jobless Claims",
            "country": "US",
            "importance": "Medium",
            "actual": None,
            "forecast": "215K",
            "previous": "217K"
        },
        {
            "date": "2023-11-10",
            "time": "10:00",
            "event": "Michigan Consumer Sentiment",
            "country": "US",
            "importance": "Medium",
            "actual": None,
            "forecast": "63.7",
            "previous": "63.8"
        },
        {
            "date": "2023-11-14",
            "time": "08:30",
            "event": "CPI m/m",
            "country": "US",
            "importance": "High",
            "actual": None,
            "forecast": "0.1%",
            "previous": "0.4%"
        },
        {
            "date": "2023-11-15",
            "time": "08:30",
            "event": "Retail Sales m/m",
            "country": "US",
            "importance": "High",
            "actual": None,
            "forecast": "0.3%",
            "previous": "0.7%"
        }
    ]
    
    # Convert to DataFrame
    df_calendar = pd.DataFrame(calendar_data)
    
    # Ensure dates are in datetime format
    df_calendar['date'] = pd.to_datetime(df_calendar['date'])
    
    # Filter by date range
    filtered_calendar = df_calendar[
        (df_calendar['date'] >= pd.to_datetime(start_date)) & 
        (df_calendar['date'] <= pd.to_datetime(end_date))
    ]
    
    if not filtered_calendar.empty:
        # Group by date
        grouped_dates = filtered_calendar.groupby('date')
        
        # Display each date's events
        for date, events in grouped_dates:
            formatted_date = date.strftime('%A, %B %d, %Y')
            st.subheader(formatted_date)
            
            # Create a table for events
            events_table = events[['time', 'event', 'country', 'importance', 'forecast', 'previous', 'actual']]
            
            # Apply styling
            def color_importance(val):
                if val == 'High':
                    return 'background-color: rgba(255, 0, 0, 0.2)'
                elif val == 'Medium':
                    return 'background-color: rgba(255, 165, 0, 0.2)'
                else:
                    return 'background-color: rgba(0, 128, 0, 0.2)'
            
            # Display table
            st.dataframe(events_table, use_container_width=True)
            st.markdown("---")
    else:
        st.info("No economic events found for the selected date range.")

    # Add disclaimer
    st.markdown("""
    **Note**: The economic calendar shown here is for demonstration purposes and may not reflect actual events. 
    In a real application, this data would come from an economic calendar API.
    """)

# Footer
st.markdown("---")
st.markdown("Stock Market Analysis Platform | Market News")
