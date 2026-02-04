"""
Generate publication-quality figures for IEEE paper
Produces forecast_comparison.png and architecture.png
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
from predictive_core import (
    arima_forecast,
    fit_prophet, forecast_prophet,
    hybrid_lstm_prophet_forecast
)

def generate_forecast_comparison(symbol='AAPL', period='6mo', forecast_days=30):
    """
    Generate a comparison chart of actual prices vs multiple forecasting models
    
    Args:
        symbol: Stock ticker symbol
        period: Historical data period
        forecast_days: Number of days to forecast
    """
    print(f"Fetching data for {symbol}...")
    
    # Fetch historical data with retry
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        print("Creating synthetic example data for demonstration...")
        # Create synthetic data for demonstration
        dates = pd.date_range(end=pd.Timestamp.now(), periods=120, freq='B')
        np.random.seed(42)
        price = 150 + np.cumsum(np.random.randn(120) * 2)
        hist = pd.DataFrame({'Close': price}, index=dates)
    
    if hist.empty:
        print("No data available! Creating synthetic data...")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=120, freq='B')
        np.random.seed(42)
        price = 150 + np.cumsum(np.random.randn(120) * 2)
        hist = pd.DataFrame({'Close': price}, index=dates)
    
    # Prepare data
    series = hist['Close'].dropna()
    
    # Split into train/test for validation
    split_idx = int(len(series) * 0.8)
    train_series = series[:split_idx]
    test_series = series[split_idx:]
    
    print(f"Training on {len(train_series)} points, testing on {len(test_series)} points")
    
    # Generate forecasts
    forecasts = {}
    
    # ARIMA
    print("Running ARIMA...")
    try:
        # arima_forecast expects DataFrame, not Series
        df_train = pd.DataFrame({'Close': train_series})
        arima_pred = arima_forecast(df=df_train, steps=len(test_series), order=(5, 1, 0))
        forecasts['ARIMA'] = arima_pred.values
    except Exception as e:
        print(f"ARIMA failed: {e}")
        forecasts['ARIMA'] = None
    
    # Prophet
    print("Running Prophet...")
    try:
        prophet_model, _ = fit_prophet(train_series)  # fit_prophet returns (model, forecast)
        prophet_pred = forecast_prophet(
            prophet_model, 
            periods=len(test_series),
            last_date=train_series.index[-1],
            freq='B'
        )
        forecasts['Prophet'] = prophet_pred.values
    except Exception as e:
        print(f"Prophet failed: {e}")
        forecasts['Prophet'] = None
    
    # Hybrid LSTM-Prophet
    print("Running Hybrid LSTM-Prophet...")
    try:
        # hybrid_lstm_prophet_forecast expects DataFrame, not Series
        df_train = pd.DataFrame({'Close': train_series})
        prophet_fc, res_fc, hybrid_pred = hybrid_lstm_prophet_forecast(
            df=df_train,
            steps=len(test_series),
            lookback=20,
            epochs=10
        )
        forecasts['Hybrid LSTM-Prophet'] = hybrid_pred.values
    except Exception as e:
        print(f"Hybrid failed: {e}")
        forecasts['Hybrid LSTM-Prophet'] = None
    
    # Create publication-quality plot
    print("Generating plot...")
    plt.figure(figsize=(12, 7))
    
    # Plot actual prices (full history)
    plt.plot(series.index, series.values, 
             label='Actual Prices', color='black', linewidth=2, alpha=0.8)
    
    # Plot test period with different style
    plt.plot(test_series.index, test_series.values,
             label='Test Period (Actual)', color='darkgreen', 
             linewidth=2.5, linestyle='--', marker='o', markersize=4)
    
    # Plot forecasts
    colors = {
        'ARIMA': '#1f77b4',
        'Prophet': '#ff7f0e', 
        'Hybrid LSTM-Prophet': '#d62728'
    }
    
    markers = {
        'ARIMA': 's',
        'Prophet': '^',
        'Hybrid LSTM-Prophet': 'D'
    }
    
    for model_name, pred in forecasts.items():
        if pred is not None and len(pred) > 0:
            # Align predictions with test dates
            pred_dates = test_series.index[:len(pred)]
            plt.plot(pred_dates, pred,
                    label=f'{model_name} Forecast',
                    color=colors.get(model_name, 'gray'),
                    linewidth=1.8,
                    marker=markers.get(model_name, 'x'),
                    markersize=6,
                    alpha=0.9)
    
    # Formatting
    plt.xlabel('Date', fontsize=14, fontweight='bold')
    plt.ylabel('Stock Price (USD)', fontsize=14, fontweight='bold')
    plt.title(f'{symbol}: 30-Day Forecast Comparison\nActual vs Predicted Prices',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=11, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save high-resolution figure
    output_file = 'forecast_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    # Calculate and print metrics
    print("\n" + "="*60)
    print("FORECAST ACCURACY METRICS")
    print("="*60)
    
    for model_name, pred in forecasts.items():
        if pred is not None and len(pred) > 0:
            # Align for comparison
            n = min(len(pred), len(test_series))
            actual = test_series.values[:n]
            predicted = pred[:n]
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((actual - predicted)**2))
            mae = np.mean(np.abs(actual - predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            # Directional accuracy
            actual_dir = np.diff(actual) > 0
            pred_dir = np.diff(predicted) > 0
            dir_acc = np.mean(actual_dir == pred_dir) * 100
            
            print(f"\n{model_name}:")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE:  {mae:.2f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  Directional Accuracy: {dir_acc:.1f}%")
    
    print("\n" + "="*60)


def generate_architecture_diagram():
    """
    Generate a system architecture diagram
    """
    print("\nGenerating architecture diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Multi-Modal Stock Prediction System Architecture',
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Layer definitions
    layers = [
        # Data Acquisition Layer
        {
            'y': 10,
            'boxes': [
                {'x': 2, 'label': 'Yahoo Finance\nAPI', 'color': '#e8f4f8'},
                {'x': 5, 'label': 'News API\n(Sentiment)', 'color': '#e8f4f8'},
                {'x': 8, 'label': 'YouTube API\n(Engagement)', 'color': '#e8f4f8'}
            ],
            'title': 'Data Acquisition Layer'
        },
        # Preprocessing
        {
            'y': 8,
            'boxes': [
                {'x': 3.5, 'label': 'Timezone\nNormalization', 'color': '#fff4e6'},
                {'x': 6.5, 'label': 'Missing Value\nImputation', 'color': '#fff4e6'}
            ],
            'title': 'Preprocessing Module'
        },
        # Forecasting Models
        {
            'y': 6,
            'boxes': [
                {'x': 1.5, 'label': 'ARIMA', 'color': '#f0f8ff'},
                {'x': 3.5, 'label': 'Prophet', 'color': '#f0f8ff'},
                {'x': 5.5, 'label': 'LSTM\nResidual', 'color': '#f0f8ff'},
                {'x': 7.5, 'label': 'Hybrid\nLSTM-Prophet', 'color': '#d4edda'}
            ],
            'title': 'Forecasting Engine'
        },
        # Analysis Modules
        {
            'y': 4,
            'boxes': [
                {'x': 2, 'label': 'Technical\nIndicators\n(RSI,MACD)', 'color': '#ffeaa7'},
                {'x': 5, 'label': 'Sentiment\nAnalysis\n(VADER)', 'color': '#ffeaa7'},
                {'x': 8, 'label': 'Anomaly\nDetection\n(IForest)', 'color': '#ffeaa7'}
            ],
            'title': 'Multi-Modal Analysis'
        },
        # HMCD Decision
        {
            'y': 2,
            'boxes': [
                {'x': 5, 'label': 'HMCD Decision Engine\nPS=0.40 | TC=0.30 | IS=0.20 | RP=0.10\nBUY / HOLD / SELL', 'color': '#ff6b6b', 'width': 3.5}
            ],
            'title': 'Decision Layer'
        },
        # UI
        {
            'y': 0.5,
            'boxes': [
                {'x': 5, 'label': 'Streamlit Web Interface\nTabbed Model Views | Real-time Updates', 'color': '#95e1d3', 'width': 4}
            ],
            'title': 'Visualization Layer'
        }
    ]
    
    # Draw layers
    for layer in layers:
        # Draw title
        ax.text(0.2, layer['y'] + 0.5, layer['title'],
                fontsize=10, fontweight='bold', style='italic',
                va='center', ha='left')
        
        # Draw boxes
        for box in layer['boxes']:
            width = box.get('width', 1.2)
            height = 0.6
            x = box['x'] - width/2
            y = layer['y'] - height/2
            
            rect = plt.Rectangle((x, y), width, height,
                                facecolor=box['color'],
                                edgecolor='black',
                                linewidth=1.5)
            ax.add_patch(rect)
            
            ax.text(box['x'], layer['y'], box['label'],
                   ha='center', va='center',
                   fontsize=8, fontweight='bold')
    
    # Draw arrows between layers
    arrow_props = dict(arrowstyle='->', lw=2, color='gray', alpha=0.6)
    
    # Data Acquisition -> Preprocessing
    ax.annotate('', xy=(5, 8.3), xytext=(5, 9.7),
                arrowprops=arrow_props)
    
    # Preprocessing -> Forecasting
    ax.annotate('', xy=(5, 6.3), xytext=(5, 7.7),
                arrowprops=arrow_props)
    
    # Forecasting -> Analysis
    ax.annotate('', xy=(5, 4.3), xytext=(5, 5.7),
                arrowprops=arrow_props)
    
    # Analysis -> HMCD
    ax.annotate('', xy=(5, 2.3), xytext=(5, 3.7),
                arrowprops=arrow_props)
    
    # HMCD -> UI
    ax.annotate('', xy=(5, 0.8), xytext=(5, 1.7),
                arrowprops=arrow_props)
    
    plt.tight_layout()
    output_file = 'architecture.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("IEEE PAPER FIGURE GENERATION")
    print("="*60)
    
    # Generate forecast comparison
    print("\n[1/2] Generating forecast comparison chart...")
    generate_forecast_comparison(symbol='AAPL', period='6mo', forecast_days=30)
    
    # Generate architecture diagram
    print("\n[2/2] Generating architecture diagram...")
    generate_architecture_diagram()
    
    print("\n" + "="*60)
    print("✓ All figures generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  • forecast_comparison.png (for experimental results section)")
    print("  • architecture.png (for system architecture section)")
    print("\nYou can now compile your LaTeX paper (h.tex) with these figures.")
