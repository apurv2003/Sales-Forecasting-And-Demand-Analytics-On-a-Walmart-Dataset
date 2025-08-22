"""
Utility functions for sales forecasting project
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Data preprocessing utilities"""
    
    @staticmethod
    def load_and_clean_data(sales_path, store_path=None):
        """Load and clean the sales data"""
        # Load sales data
        sales_df = pd.read_csv(sales_path)
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        
        # Load store features if provided
        store_df = None
        if store_path:
            store_df = pd.read_csv(store_path)
        
        return sales_df, store_df
    
    @staticmethod
    def aggregate_by_date(sales_df, group_cols=['Date'], agg_col='Weekly_Sales'):
        """Aggregate sales by date"""
        return sales_df.groupby(group_cols)[agg_col].sum().reset_index()
    
    @staticmethod
    def create_time_features(df, date_col='Date'):
        """Create time-based features"""
        df = df.copy()
        df['Year'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month
        df['Week'] = df[date_col].dt.isocalendar().week
        df['DayOfWeek'] = df[date_col].dt.dayofweek
        df['Quarter'] = df[date_col].dt.quarter
        df['DayOfYear'] = df[date_col].dt.dayofyear
        
        return df
    
    @staticmethod
    def add_lag_features(df, target_col='Weekly_Sales', lags=[1, 2, 4, 8, 12]):
        """Add lag features for time series"""
        df = df.copy()
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        return df
    
    @staticmethod
    def add_rolling_features(df, target_col='Weekly_Sales', windows=[7, 14, 30]):
        """Add rolling window features"""
        df = df.copy()
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        return df

class EvaluationMetrics:
    """Evaluation metrics for forecasting models"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate various evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    @staticmethod
    def plot_predictions(y_true, y_pred, title='Actual vs Predicted'):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_residuals(y_true, y_pred, title='Residuals Plot'):
        """Plot residuals"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals over time
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=30, alpha=0.7)
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Residuals vs Predicted
        axes[1, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title('Residuals vs Predicted')
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        
        plt.tight_layout()
        plt.show()

class VisualizationUtils:
    """Visualization utilities"""
    
    @staticmethod
    def plot_sales_trends(df, date_col='Date', sales_col='Weekly_Sales'):
        """Plot sales trends over time"""
        plt.figure(figsize=(15, 8))
        plt.plot(df[date_col], df[sales_col])
        plt.title('Sales Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Weekly Sales')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_seasonality(df, date_col='Date', sales_col='Weekly_Sales'):
        """Plot seasonal patterns"""
        df = df.copy()
        df['Month'] = df[date_col].dt.month
        df['Year'] = df[date_col].dt.year
        
        # Monthly seasonality
        monthly_avg = df.groupby('Month')[sales_col].mean()
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        monthly_avg.plot(kind='bar')
        plt.title('Average Sales by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Sales')
        plt.xticks(rotation=0)
        
        # Yearly trend
        yearly_avg = df.groupby('Year')[sales_col].mean()
        plt.subplot(1, 2, 2)
        yearly_avg.plot(kind='line', marker='o')
        plt.title('Average Sales by Year')
        plt.xlabel('Year')
        plt.ylabel('Average Sales')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(df, numeric_cols):
        """Plot correlation matrix"""
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

class DataSplitter:
    """Time series data splitting utilities"""
    
    @staticmethod
    def train_test_split_time_series(df, date_col='Date', test_size=0.2):
        """Split data into train and test sets maintaining temporal order"""
        df = df.sort_values(date_col)
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        return train_df, test_df
    
    @staticmethod
    def create_sequences(data, target_col, sequence_length=12):
        """Create sequences for LSTM model"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[target_col].values[i:(i + sequence_length)])
            y.append(data[target_col].values[i + sequence_length])
        return np.array(X), np.array(y)

def save_model_results(model_name, metrics, predictions, filepath):
    """Save model results to file"""
    results = {
        'model_name': model_name,
        'metrics': metrics,
        'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
    }
    
    import json
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

def load_model_results(filepath):
    """Load model results from file"""
    import json
    with open(filepath, 'r') as f:
        return json.load(f) 