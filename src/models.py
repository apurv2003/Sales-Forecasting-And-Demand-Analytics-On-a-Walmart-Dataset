"""
Forecasting Models for Sales Prediction
Implements ARIMA, Prophet, and LSTM models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Time series models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")

# Deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

from src.utils import DataPreprocessor, EvaluationMetrics, DataSplitter, save_model_results

class ForecastingModels:
    """Collection of forecasting models"""
    
    def __init__(self, data_path='data/walmart_sales.csv'):
        """Initialize with data"""
        self.sales_df, _ = DataPreprocessor.load_and_clean_data(data_path)
        self.aggregated_sales = DataPreprocessor.aggregate_by_date(self.sales_df)
        self.aggregated_sales = DataPreprocessor.create_time_features(self.aggregated_sales)
        self.aggregated_sales = self.aggregated_sales.sort_values('Date').reset_index(drop=True)
        
        # Split data
        self.train_data, self.test_data = DataSplitter.train_test_split_time_series(
            self.aggregated_sales, test_size=0.2
        )
        
        # Model results storage
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
        print(f"Data loaded: {len(self.aggregated_sales)} total records")
        print(f"Training set: {len(self.train_data)} records")
        print(f"Test set: {len(self.test_data)} records")
    
    def check_stationarity(self, data, title="Time Series"):
        """Check if time series is stationary"""
        print(f"\n=== Stationarity Test for {title} ===")
        
        # Perform Augmented Dickey-Fuller test
        result = adfuller(data)
        
        print(f'ADF Statistic: {result[0]:.6f}')
        print(f'p-value: {result[1]:.6f}')
        print(f'Critical values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.3f}')
        
        if result[1] <= 0.05:
            print("✅ Series is stationary (p-value <= 0.05)")
            return True
        else:
            print("❌ Series is not stationary (p-value > 0.05)")
            return False
    
    def make_stationary(self, data, max_diff=2):
        """Make time series stationary through differencing"""
        original_data = data.copy()
        diff_data = data.copy()
        diff_order = 0
        
        for i in range(max_diff):
            if self.check_stationarity(diff_data, f"Differenced Series (order {i+1})"):
                break
            diff_data = np.diff(diff_data)
            diff_order += 1
        
        if diff_order == 0:
            print("Original series is already stationary")
            return original_data, 0
        else:
            print(f"Series made stationary with {diff_order} differencing")
            return diff_data, diff_order
    
    def train_arima(self, auto_arima=True, order=None):
        """Train ARIMA model"""
        print("\n=== Training ARIMA Model ===")
        
        # Prepare data
        train_series = self.train_data['Weekly_Sales'].values
        test_series = self.test_data['Weekly_Sales'].values
        
        # Check stationarity
        is_stationary = self.check_stationarity(train_series, "Training Data")
        
        if not is_stationary:
            print("Making series stationary...")
            train_series, diff_order = self.make_stationary(train_series)
        
        # Auto ARIMA or manual order
        if auto_arima:
            try:
                from pmdarima import auto_arima
                model = auto_arima(train_series, seasonal=False, 
                                 error_action='ignore', suppress_warnings=True,
                                 stepwise=True, max_p=3, max_q=3, max_d=2)
                order = model.order
                print(f"Auto ARIMA selected order: {order}")
            except ImportError:
                print("pmdarima not available. Using default order (1,1,1)")
                order = (1, 1, 1)
        else:
            order = order or (1, 1, 1)
        
        # Train ARIMA model
        if not is_stationary and diff_order > 0:
            # Use differenced data
            arima_model = ARIMA(train_series, order=order)
        else:
            # Use original data
            arima_model = ARIMA(self.train_data['Weekly_Sales'], order=order)
        
        fitted_model = arima_model.fit()
        
        # Make predictions
        forecast_steps = len(self.test_data)
        forecast = fitted_model.forecast(steps=forecast_steps)
        
        # If we used differencing, we need to reverse it
        if not is_stationary and diff_order > 0:
            # This is a simplified approach - in practice, you'd need more sophisticated inverse differencing
            forecast = self._inverse_difference(forecast, self.train_data['Weekly_Sales'].iloc[-diff_order:])
        
        # Store results
        self.models['ARIMA'] = fitted_model
        self.predictions['ARIMA'] = forecast
        
        # Calculate metrics
        metrics = EvaluationMetrics.calculate_metrics(test_series, forecast)
        self.metrics['ARIMA'] = metrics
        
        print(f"ARIMA Model Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return fitted_model, forecast, metrics
    
    def _inverse_difference(self, forecast, last_values):
        """Simple inverse differencing (simplified)"""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated inverse differencing
        result = []
        for i, pred in enumerate(forecast):
            if i == 0:
                result.append(last_values.iloc[-1] + pred)
            else:
                result.append(result[-1] + pred)
        return np.array(result)
    
    def train_prophet(self):
        """Train Prophet model"""
        if not PROPHET_AVAILABLE:
            print("Prophet not available. Skipping Prophet model.")
            return None, None, None
        
        print("\n=== Training Prophet Model ===")
        
        # Prepare data for Prophet
        prophet_data = self.train_data[['Date', 'Weekly_Sales']].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Add holiday information
        holiday_data = self.sales_df[self.sales_df['IsHoliday'] == True][['Date']].copy()
        holiday_data.columns = ['ds']
        holiday_data['holiday'] = 'Holiday'
        
        # Create and train Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=holiday_data,
            seasonality_mode='multiplicative'
        )
        
        model.fit(prophet_data)
        
        # Make predictions
        future = model.make_future_dataframe(periods=len(self.test_data), freq='W')
        forecast = model.predict(future)
        
        # Extract predictions for test period
        test_predictions = forecast.iloc[-len(self.test_data):]['yhat'].values
        
        # Store results
        self.models['Prophet'] = model
        self.predictions['Prophet'] = test_predictions
        
        # Calculate metrics
        test_series = self.test_data['Weekly_Sales'].values
        metrics = EvaluationMetrics.calculate_metrics(test_series, test_predictions)
        self.metrics['Prophet'] = metrics
        
        print(f"Prophet Model Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return model, test_predictions, metrics
    
    def train_lstm(self, sequence_length=12, epochs=50, batch_size=32):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping LSTM model.")
            return None, None, None
        
        print("\n=== Training LSTM Model ===")
        
        # Prepare data
        data = self.aggregated_sales['Weekly_Sales'].values.reshape(-1, 1)
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        X, y = DataSplitter.create_sequences(
            pd.DataFrame(scaled_data, columns=['Weekly_Sales']), 
            'Weekly_Sales', 
            sequence_length
        )
        
        # Split sequences into train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Reshape for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Inverse transform
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Align with test data
        test_series = self.test_data['Weekly_Sales'].values
        if len(predictions) > len(test_series):
            predictions = predictions[-len(test_series):]
        elif len(predictions) < len(test_series):
            # Pad with last prediction
            padding = np.full(len(test_series) - len(predictions), predictions[-1])
            predictions = np.concatenate([predictions, padding])
        
        # Store results
        self.models['LSTM'] = model
        self.predictions['LSTM'] = predictions.flatten()
        
        # Calculate metrics
        metrics = EvaluationMetrics.calculate_metrics(test_series, predictions.flatten())
        self.metrics['LSTM'] = metrics
        
        print(f"LSTM Model Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return model, predictions.flatten(), metrics
    
    def compare_models(self):
        """Compare all trained models"""
        if not self.metrics:
            print("No models trained yet. Train models first.")
            return
        
        print("\n=== Model Comparison ===")
        
        # Create comparison table
        comparison_df = pd.DataFrame(self.metrics).T
        print(comparison_df.round(4))
        
        # Plot comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RMSE Comparison', 'MAE Comparison', 
                          'MAPE Comparison', 'Predictions vs Actual'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Metrics comparison
        metrics_to_plot = ['RMSE', 'MAE', 'MAPE']
        colors = ['blue', 'red', 'green']
        
        for i, metric in enumerate(metrics_to_plot):
            fig.add_trace(
                go.Bar(x=list(self.metrics.keys()), 
                      y=[self.metrics[model][metric] for model in self.metrics.keys()],
                      name=metric, marker_color=colors[i]),
                row=1, col=i+1
            )
        
        # Predictions comparison
        test_series = self.test_data['Weekly_Sales'].values
        fig.add_trace(
            go.Scatter(x=range(len(test_series)), y=test_series,
                      mode='lines', name='Actual', line=dict(color='black')),
            row=2, col=1
        )
        
        for model_name, predictions in self.predictions.items():
            fig.add_trace(
                go.Scatter(x=range(len(predictions)), y=predictions,
                          mode='lines', name=f'{model_name} Pred', line=dict(dash='dash')),
                row=2, col=1
            )
        
        fig.update_layout(height=800, title_text="Model Comparison")
        fig.show()
        
        return comparison_df
    
    def plot_predictions(self, model_name=None):
        """Plot predictions for specific model or all models"""
        if model_name:
            models_to_plot = [model_name]
        else:
            models_to_plot = list(self.predictions.keys())
        
        test_series = self.test_data['Weekly_Sales'].values
        test_dates = self.test_data['Date'].values
        
        fig = go.Figure()
        
        # Plot actual values
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_series,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=3)
        ))
        
        # Plot predictions
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, model in enumerate(models_to_plot):
            if model in self.predictions:
                fig.add_trace(go.Scatter(
                    x=test_dates,
                    y=self.predictions[model],
                    mode='lines',
                    name=f'{model} Prediction',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
        
        fig.update_layout(
            title=f'Sales Predictions: {", ".join(models_to_plot)}',
            xaxis_title='Date',
            yaxis_title='Weekly Sales ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        fig.show()
    
    def save_all_results(self, output_dir='results'):
        """Save all model results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name in self.models.keys():
            if model_name in self.predictions and model_name in self.metrics:
                filepath = os.path.join(output_dir, f'{model_name.lower()}_results.json')
                save_model_results(
                    model_name,
                    self.metrics[model_name],
                    self.predictions[model_name],
                    filepath
                )
                print(f"Saved {model_name} results to {filepath}")
    
    def get_best_model(self):
        """Get the best performing model based on RMSE"""
        if not self.metrics:
            return None
        
        best_model = min(self.metrics.keys(), 
                        key=lambda x: self.metrics[x]['RMSE'])
        best_rmse = self.metrics[best_model]['RMSE']
        
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   RMSE: {best_rmse:.4f}")
        
        return best_model

def main():
    """Main function to train all models"""
    # Initialize models
    forecaster = ForecastingModels()
    
    # Train models
    print("Training forecasting models...")
    
    # ARIMA
    forecaster.train_arima()
    
    # Prophet
    forecaster.train_prophet()
    
    # LSTM
    forecaster.train_lstm()
    
    # Compare models
    comparison = forecaster.compare_models()
    
    # Plot predictions
    forecaster.plot_predictions()
    
    # Get best model
    best_model = forecaster.get_best_model()
    
    # Save results
    forecaster.save_all_results()
    
    print("\nModel training completed!")

if __name__ == "__main__":
    main() 