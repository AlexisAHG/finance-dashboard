"""
Finance Dashboard - Main Application
=====================================
A professional financial dashboard for real-time market analysis,
backtesting strategies, and portfolio management.

Authors: Quant A & Quant B
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Import custom modules
from modules.single_asset import SingleAssetAnalysis
from modules.portfolio import PortfolioAnalysis
from utils.data_fetcher import DataFetcher
from utils.metrics import FinancialMetrics

# Page configuration
st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    .refresh-info {
        font-size: 0.8rem;
        color: #888;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

def main():
    """Main application function."""
    
    # Header
    st.markdown('<p class="main-header">📊 Finance Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time Market Analysis & Portfolio Management</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/stock-share.png", width=80)
        st.title("⚙️ Configuration")
        
        # Auto-refresh toggle
        auto_refresh = st.toggle("🔄 Auto-refresh (5 min)", value=True)
        
        if auto_refresh:
            # Check if 5 minutes have passed
            time_diff = (datetime.now() - st.session_state.last_refresh).seconds
            if time_diff >= 300:  # 5 minutes = 300 seconds
                st.session_state.last_refresh = datetime.now()
                st.session_state.data_cache = {}
                st.rerun()
            
            # Display countdown
            remaining = 300 - time_diff
            mins, secs = divmod(remaining, 60)
            st.info(f"⏱️ Next refresh in: {mins:02d}:{secs:02d}")
        
        # Manual refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.session_state.last_refresh = datetime.now()
            st.session_state.data_cache = {}
            st.rerun()
        
        st.divider()
        
        # Last update time
        st.markdown(f"**Last Update:**")
        st.markdown(f"📅 {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.divider()
        
        # Data source selection
        st.subheader("📡 Data Source")
        data_source = st.selectbox(
            "Select API",
            ["Yahoo Finance", "Alpha Vantage", "CoinGecko (Crypto)"],
            help="Choose your preferred data source"
        )
        
        st.divider()
        
        # Information
        st.subheader("ℹ️ About")
        st.markdown("""
        This dashboard provides:
        - 📈 Real-time market data
        - 🎯 Backtesting strategies
        - 💼 Portfolio analysis
        - 📊 Performance metrics
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "📈 Single Asset Analysis (Quant A)",
        "💼 Portfolio Analysis (Quant B)", 
        "📋 Daily Reports"
    ])
    
    # Initialize data fetcher
    fetcher = DataFetcher()
    
    # Tab 1: Single Asset Analysis (Quant A)
    with tab1:
        single_asset_module(fetcher)
    
    # Tab 2: Portfolio Analysis (Quant B)
    with tab2:
        portfolio_module(fetcher)
    
    # Tab 3: Daily Reports
    with tab3:
        reports_module()
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown(
            "<p style='text-align: center; color: #888;'>"
            "Finance Dashboard v1.0 | Made with ❤️ using Streamlit"
            "</p>",
            unsafe_allow_html=True
        )


def single_asset_module(fetcher):
    """Single Asset Analysis Module (Quant A)."""
    
    st.header("📈 Single Asset Analysis")
    st.markdown("Analyze individual assets with backtesting strategies and performance metrics.")
    
    # Asset selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        asset_options = {
            "Stocks": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"],
            "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"],
            "Commodities": ["GC=F", "SI=F", "CL=F"],  # Gold, Silver, Oil
            "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "French Stocks": ["ENGI.PA", "TTE.PA", "AIR.PA", "BNP.PA", "SAN.PA"]
        }
        
        asset_category = st.selectbox("Asset Category", list(asset_options.keys()))
        selected_asset = st.selectbox("Select Asset", asset_options[asset_category])
    
    with col2:
        period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
    
    with col3:
        interval = st.selectbox(
            "Interval",
            ["1d", "1wk", "1mo"],
            index=0
        )
    
    # Fetch data
    try:
        with st.spinner(f"Fetching data for {selected_asset}..."):
            df = fetcher.get_stock_data(selected_asset, period=period, interval=interval)
        
        if df is not None and not df.empty:
            # Display current price metrics
            st.subheader("📊 Current Market Data")
            
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{price_change_pct:+.2f}%"
                )
            
            with col2:
                st.metric("Open", f"${df['Open'].iloc[-1]:.2f}")
            
            with col3:
                st.metric("High", f"${df['High'].iloc[-1]:.2f}")
            
            with col4:
                st.metric("Low", f"${df['Low'].iloc[-1]:.2f}")
            
            with col5:
                vol = df['Volume'].iloc[-1]
                vol_str = f"{vol/1e6:.2f}M" if vol >= 1e6 else f"{vol/1e3:.2f}K"
                st.metric("Volume", vol_str)
            
            st.divider()
            
            # Strategy selection
            st.subheader("🎯 Backtesting Strategies")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                strategy = st.selectbox(
                    "Select Strategy",
                    ["Buy and Hold", "Moving Average Crossover", "RSI Strategy", "Momentum"]
                )
                
                # Strategy parameters
                if strategy == "Moving Average Crossover":
                    short_window = st.slider("Short MA Window", 5, 50, 20)
                    long_window = st.slider("Long MA Window", 20, 200, 50)
                elif strategy == "RSI Strategy":
                    rsi_period = st.slider("RSI Period", 5, 30, 14)
                    rsi_oversold = st.slider("Oversold Level", 20, 40, 30)
                    rsi_overbought = st.slider("Overbought Level", 60, 80, 70)
                elif strategy == "Momentum":
                    momentum_period = st.slider("Momentum Period", 5, 60, 20)
                
                initial_capital = st.number_input(
                    "Initial Capital ($)",
                    min_value=1000,
                    max_value=1000000,
                    value=10000,
                    step=1000
                )
            
            # Initialize analysis
            analyzer = SingleAssetAnalysis(df, initial_capital)
            
            # Run selected strategy
            if strategy == "Buy and Hold":
                strategy_df = analyzer.buy_and_hold()
            elif strategy == "Moving Average Crossover":
                strategy_df = analyzer.ma_crossover(short_window, long_window)
            elif strategy == "RSI Strategy":
                strategy_df = analyzer.rsi_strategy(rsi_period, rsi_oversold, rsi_overbought)
            else:  # Momentum
                strategy_df = analyzer.momentum_strategy(momentum_period)
            
            with col2:
                # Performance metrics
                metrics = FinancialMetrics.calculate_metrics(strategy_df, initial_capital)
                
                st.markdown("**📈 Performance Metrics**")
                
                met_col1, met_col2, met_col3 = st.columns(3)
                
                with met_col1:
                    st.metric(
                        "Total Return",
                        f"{metrics['total_return']:.2f}%",
                        delta=None
                    )
                    st.metric(
                        "Max Drawdown",
                        f"{metrics['max_drawdown']:.2f}%"
                    )
                
                with met_col2:
                    st.metric(
                        "Sharpe Ratio",
                        f"{metrics['sharpe_ratio']:.3f}"
                    )
                    st.metric(
                        "Volatility (Ann.)",
                        f"{metrics['volatility']:.2f}%"
                    )
                
                with met_col3:
                    st.metric(
                        "Final Value",
                        f"${metrics['final_value']:,.2f}"
                    )
                    st.metric(
                        "Win Rate",
                        f"{metrics.get('win_rate', 0):.1f}%"
                    )
            
            # Main chart with price and strategy
            st.subheader("📉 Price & Strategy Performance")
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{selected_asset} Price & Strategy', 'Volume')
            )
            
            # Price line
            fig.add_trace(
                go.Scatter(
                    x=strategy_df.index,
                    y=strategy_df['Close'],
                    name='Asset Price',
                    line=dict(color='#2962FF', width=2)
                ),
                row=1, col=1
            )
            
            # Strategy cumulative value (normalized to price scale for comparison)
            normalized_strategy = strategy_df['Strategy_Value'] / strategy_df['Strategy_Value'].iloc[0] * strategy_df['Close'].iloc[0]
            
            fig.add_trace(
                go.Scatter(
                    x=strategy_df.index,
                    y=normalized_strategy,
                    name=f'{strategy} Strategy',
                    line=dict(color='#00C853', width=2)
                ),
                row=1, col=1
            )
            
            # Add moving averages if applicable
            if strategy == "Moving Average Crossover":
                if 'SMA_Short' in strategy_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=strategy_df.index,
                            y=strategy_df['SMA_Short'],
                            name=f'SMA {short_window}',
                            line=dict(color='orange', width=1, dash='dash')
                        ),
                        row=1, col=1
                    )
                if 'SMA_Long' in strategy_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=strategy_df.index,
                            y=strategy_df['SMA_Long'],
                            name=f'SMA {long_window}',
                            line=dict(color='purple', width=1, dash='dash')
                        ),
                        row=1, col=1
                    )
            
            # Volume bars
            colors = ['#EF5350' if strategy_df['Close'].iloc[i] < strategy_df['Open'].iloc[i] 
                     else '#26A69A' for i in range(len(strategy_df))]
            
            fig.add_trace(
                go.Bar(
                    x=strategy_df.index,
                    y=strategy_df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis_rangeslider_visible=False,
                template="plotly_white"
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Strategy value chart
            st.subheader("💰 Portfolio Value Over Time")
            
            fig_value = go.Figure()
            
            fig_value.add_trace(
                go.Scatter(
                    x=strategy_df.index,
                    y=strategy_df['Strategy_Value'],
                    fill='tozeroy',
                    name='Strategy Value',
                    line=dict(color='#00C853', width=2),
                    fillcolor='rgba(0, 200, 83, 0.2)'
                )
            )
            
            # Buy and Hold comparison
            buy_hold_value = initial_capital * (df['Close'] / df['Close'].iloc[0])
            fig_value.add_trace(
                go.Scatter(
                    x=df.index,
                    y=buy_hold_value,
                    name='Buy & Hold',
                    line=dict(color='#FF6D00', width=2, dash='dash')
                )
            )
            
            fig_value.update_layout(
                height=400,
                showlegend=True,
                template="plotly_white",
                yaxis_title="Portfolio Value ($)",
                xaxis_title="Date"
            )
            
            st.plotly_chart(fig_value, use_container_width=True)
            
            # Additional analysis
            with st.expander("📊 Detailed Statistics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Price Statistics**")
                    stats_df = pd.DataFrame({
                        'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median'],
                        'Value': [
                            f"${df['Close'].mean():.2f}",
                            f"${df['Close'].std():.2f}",
                            f"${df['Close'].min():.2f}",
                            f"${df['Close'].max():.2f}",
                            f"${df['Close'].median():.2f}"
                        ]
                    })
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)
                
                with col2:
                    st.markdown("**Returns Statistics**")
                    returns = df['Close'].pct_change().dropna()
                    ret_stats_df = pd.DataFrame({
                        'Metric': ['Mean Daily', 'Std Daily', 'Skewness', 'Kurtosis', 'Best Day', 'Worst Day'],
                        'Value': [
                            f"{returns.mean()*100:.3f}%",
                            f"{returns.std()*100:.3f}%",
                            f"{returns.skew():.3f}",
                            f"{returns.kurtosis():.3f}",
                            f"{returns.max()*100:.2f}%",
                            f"{returns.min()*100:.2f}%"
                        ]
                    })
                    st.dataframe(ret_stats_df, hide_index=True, use_container_width=True)
        
        else:
            st.error("Unable to fetch data. Please try again or select a different asset.")
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check your internet connection or try a different asset.")


def portfolio_module(fetcher):
    """Portfolio Analysis Module (Quant B)."""
    
    st.header("💼 Multi-Asset Portfolio Analysis")
    st.markdown("Build and analyze diversified portfolios with multiple assets.")
    
    # Portfolio configuration
    st.subheader("🔧 Portfolio Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Predefined portfolios or custom
        portfolio_type = st.radio(
            "Portfolio Type",
            ["Predefined Portfolios", "Custom Selection"],
            horizontal=True
        )
        
        if portfolio_type == "Predefined Portfolios":
            predefined = st.selectbox(
                "Select Portfolio",
                [
                    "Tech Giants (AAPL, GOOGL, MSFT, NVDA)",
                    "Diversified (AAPL, GC=F, BTC-USD, EURUSD=X)",
                    "French CAC40 (ENGI.PA, TTE.PA, AIR.PA, BNP.PA)",
                    "Crypto Mix (BTC-USD, ETH-USD, SOL-USD)"
                ]
            )
            
            portfolio_map = {
                "Tech Giants (AAPL, GOOGL, MSFT, NVDA)": ["AAPL", "GOOGL", "MSFT", "NVDA"],
                "Diversified (AAPL, GC=F, BTC-USD, EURUSD=X)": ["AAPL", "GC=F", "BTC-USD", "EURUSD=X"],
                "French CAC40 (ENGI.PA, TTE.PA, AIR.PA, BNP.PA)": ["ENGI.PA", "TTE.PA", "AIR.PA", "BNP.PA"],
                "Crypto Mix (BTC-USD, ETH-USD, SOL-USD)": ["BTC-USD", "ETH-USD", "SOL-USD"]
            }
            selected_assets = portfolio_map[predefined]
        else:
            all_assets = [
                "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META",
                "ENGI.PA", "TTE.PA", "AIR.PA", "BNP.PA", "SAN.PA",
                "GC=F", "SI=F", "CL=F",
                "BTC-USD", "ETH-USD", "SOL-USD",
                "EURUSD=X", "GBPUSD=X"
            ]
            selected_assets = st.multiselect(
                "Select Assets (min 3)",
                all_assets,
                default=["AAPL", "GOOGL", "MSFT"]
            )
    
    with col2:
        period = st.selectbox(
            "Analysis Period",
            ["3mo", "6mo", "1y", "2y"],
            index=2,
            key="portfolio_period"
        )
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=5000,
            key="portfolio_capital"
        )
    
    if len(selected_assets) < 3:
        st.warning("⚠️ Please select at least 3 assets for portfolio analysis.")
        return
    
    # Weight allocation
    st.subheader("⚖️ Weight Allocation")
    
    weight_method = st.radio(
        "Allocation Method",
        ["Equal Weight", "Custom Weights", "Risk Parity (Inverse Volatility)"],
        horizontal=True
    )
    
    weights = {}
    
    if weight_method == "Equal Weight":
        equal_weight = 1.0 / len(selected_assets)
        for asset in selected_assets:
            weights[asset] = equal_weight
        
        # Display weights
        weight_df = pd.DataFrame({
            'Asset': selected_assets,
            'Weight': [f"{equal_weight*100:.1f}%" for _ in selected_assets]
        })
        st.dataframe(weight_df, hide_index=True, use_container_width=True)
    
    elif weight_method == "Custom Weights":
        cols = st.columns(len(selected_assets))
        total_weight = 0
        
        for i, asset in enumerate(selected_assets):
            with cols[i]:
                w = st.slider(
                    asset,
                    0, 100, int(100/len(selected_assets)),
                    key=f"weight_{asset}"
                )
                weights[asset] = w / 100
                total_weight += w
        
        if abs(total_weight - 100) > 0.1:
            st.warning(f"⚠️ Total weight: {total_weight}% (should be 100%)")
        else:
            st.success(f"✅ Total weight: {total_weight}%")
    
    # Fetch portfolio data
    try:
        with st.spinner("Fetching portfolio data..."):
            portfolio_data = {}
            for asset in selected_assets:
                df = fetcher.get_stock_data(asset, period=period, interval="1d")
                if df is not None and not df.empty:
                    portfolio_data[asset] = df['Close']
            
            if len(portfolio_data) < 3:
                st.error("Could not fetch data for enough assets. Please try different selections.")
                return
            
            # Create combined DataFrame
            portfolio_df = pd.DataFrame(portfolio_data)
            portfolio_df = portfolio_df.dropna()
            
            # If Risk Parity, calculate weights based on inverse volatility
            if weight_method == "Risk Parity (Inverse Volatility)":
                returns = portfolio_df.pct_change().dropna()
                volatilities = returns.std()
                inv_vol = 1 / volatilities
                weights = (inv_vol / inv_vol.sum()).to_dict()
                
                weight_df = pd.DataFrame({
                    'Asset': list(weights.keys()),
                    'Weight': [f"{w*100:.1f}%" for w in weights.values()],
                    'Volatility': [f"{v*100:.2f}%" for v in volatilities]
                })
                st.dataframe(weight_df, hide_index=True, use_container_width=True)
        
        # Initialize portfolio analyzer
        analyzer = PortfolioAnalysis(portfolio_df, weights, initial_capital)
        
        # Display current values
        st.subheader("📊 Current Asset Values")
        
        current_cols = st.columns(len(selected_assets))
        for i, asset in enumerate(selected_assets):
            with current_cols[i]:
                current = portfolio_df[asset].iloc[-1]
                prev = portfolio_df[asset].iloc[-2] if len(portfolio_df) > 1 else current
                change_pct = ((current - prev) / prev) * 100
                st.metric(asset, f"${current:.2f}", f"{change_pct:+.2f}%")
        
        st.divider()
        
        # Portfolio metrics
        st.subheader("📈 Portfolio Performance")
        
        portfolio_value, metrics = analyzer.calculate_portfolio_value()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", f"${portfolio_value.iloc[-1]:,.2f}")
            st.metric("Total Return", f"{metrics['total_return']:.2f}%")
        
        with col2:
            st.metric("Portfolio Volatility", f"{metrics['portfolio_volatility']:.2f}%")
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
        
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}")
        
        with col4:
            st.metric("Diversification Ratio", f"{metrics.get('diversification_ratio', 1):.2f}")
            st.metric("VaR (95%)", f"{metrics.get('var_95', 0):.2f}%")
        
        st.divider()
        
        # Main chart - Multiple assets + Portfolio
        st.subheader("📉 Asset Prices & Portfolio Value")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Normalize prices to 100 for comparison
        normalized_prices = portfolio_df / portfolio_df.iloc[0] * 100
        
        colors = ['#2962FF', '#00C853', '#FF6D00', '#AA00FF', '#00BCD4', '#FF5252']
        
        for i, asset in enumerate(selected_assets):
            fig.add_trace(
                go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[asset],
                    name=asset,
                    line=dict(color=colors[i % len(colors)], width=2)
                )
            )
        
        # Add portfolio line
        normalized_portfolio = portfolio_value / portfolio_value.iloc[0] * 100
        fig.add_trace(
            go.Scatter(
                x=portfolio_value.index,
                y=normalized_portfolio,
                name='Portfolio',
                line=dict(color='#FFD700', width=3)
            )
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white",
            yaxis_title="Normalized Value (Base 100)",
            xaxis_title="Date"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio value chart
        st.subheader("💰 Portfolio Value Over Time")
        
        fig_portfolio = go.Figure()
        
        fig_portfolio.add_trace(
            go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value,
                fill='tozeroy',
                name='Portfolio Value',
                line=dict(color='#00C853', width=2),
                fillcolor='rgba(0, 200, 83, 0.2)'
            )
        )
        
        fig_portfolio.update_layout(
            height=400,
            template="plotly_white",
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Date"
        )
        
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Correlation matrix
        st.subheader("🔗 Correlation Matrix")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            returns = portfolio_df.pct_change().dropna()
            corr_matrix = returns.corr()
            
            import plotly.figure_factory as ff
            
            fig_corr = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=list(corr_matrix.columns),
                y=list(corr_matrix.index),
                annotation_text=np.around(corr_matrix.values, decimals=2),
                colorscale='RdBu',
                showscale=True
            )
            
            fig_corr.update_layout(
                height=400,
                title="Asset Correlations"
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            # Weights pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=.4,
                marker_colors=colors[:len(weights)]
            )])
            
            fig_pie.update_layout(
                height=400,
                title="Portfolio Allocation"
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Rebalancing simulation
        st.subheader("🔄 Rebalancing Simulation")
        
        rebal_freq = st.selectbox(
            "Rebalancing Frequency",
            ["No Rebalancing", "Monthly", "Quarterly", "Yearly"]
        )
        
        if rebal_freq != "No Rebalancing":
            rebal_portfolio, rebal_metrics = analyzer.simulate_rebalancing(rebal_freq)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Without Rebalancing**")
                st.metric("Final Value", f"${portfolio_value.iloc[-1]:,.2f}")
                st.metric("Total Return", f"{metrics['total_return']:.2f}%")
            
            with col2:
                st.markdown(f"**With {rebal_freq} Rebalancing**")
                st.metric("Final Value", f"${rebal_portfolio.iloc[-1]:,.2f}")
                st.metric("Total Return", f"{rebal_metrics['total_return']:.2f}%")
        
        # Detailed statistics
        with st.expander("📊 Detailed Asset Statistics"):
            stats_data = []
            for asset in selected_assets:
                returns = portfolio_df[asset].pct_change().dropna()
                stats_data.append({
                    'Asset': asset,
                    'Mean Return': f"{returns.mean()*252*100:.2f}%",
                    'Volatility': f"{returns.std()*np.sqrt(252)*100:.2f}%",
                    'Sharpe': f"{(returns.mean()/returns.std())*np.sqrt(252):.2f}",
                    'Max DD': f"{((portfolio_df[asset].cummax() - portfolio_df[asset])/portfolio_df[asset].cummax()).max()*100:.2f}%",
                    'Weight': f"{weights[asset]*100:.1f}%"
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error in portfolio analysis: {str(e)}")
        st.info("Please try selecting different assets or check your connection.")


def reports_module():
    """Daily Reports Module."""
    
    st.header("📋 Daily Reports")
    st.markdown("View automatically generated daily reports (updated via cron at 8 PM).")
    
    # Report viewer
    import os
    reports_dir = "reports"
    
    if os.path.exists(reports_dir):
        reports = sorted([f for f in os.listdir(reports_dir) if f.endswith('.txt')], reverse=True)
        
        if reports:
            selected_report = st.selectbox("Select Report", reports)
            
            with open(os.path.join(reports_dir, selected_report), 'r') as f:
                report_content = f.read()
            
            st.code(report_content, language="text")
        else:
            st.info("No reports generated yet. Reports are created daily at 8 PM.")
    else:
        st.info("Reports directory not found. Make sure the cron job is configured.")
    
    # Manual report generation
    st.divider()
    st.subheader("🔧 Manual Report Generation")
    
    if st.button("Generate Report Now", use_container_width=True):
        with st.spinner("Generating report..."):
            try:
                from utils.report_generator import ReportGenerator
                generator = ReportGenerator()
                report_path = generator.generate_daily_report()
                st.success(f"Report generated: {report_path}")
                st.rerun()
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    # Cron job information
    with st.expander("ℹ️ Cron Job Configuration"):
        st.markdown("""
        The daily report is generated automatically using a cron job.
        
        **Cron Configuration:**
        ```bash
        0 20 * * * /path/to/venv/bin/python /path/to/scripts/generate_report.py
        ```
        
        **Report Contents:**
        - Daily price changes for tracked assets
        - Volatility metrics
        - Max drawdown calculations
        - Open/Close prices
        - Volume statistics
        """)


if __name__ == "__main__":
    main()
