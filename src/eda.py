"""
Exploratory Data Analysis for Sales Forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils import DataPreprocessor, VisualizationUtils
import warnings
warnings.filterwarnings('ignore')

class SalesAnalyzer:
    """Comprehensive sales data analyzer"""
    
    def __init__(self, sales_path, store_path=None):
        """Initialize with data paths"""
        self.sales_df, self.store_df = DataPreprocessor.load_and_clean_data(sales_path, store_path)
        self.aggregated_sales = None
        self.analyze_data()
    
    def analyze_data(self):
        """Perform initial data analysis"""
        print("=== Sales Data Overview ===")
        print(f"Total records: {len(self.sales_df):,}")
        print(f"Date range: {self.sales_df['Date'].min()} to {self.sales_df['Date'].max()}")
        print(f"Number of stores: {self.sales_df['Store'].nunique()}")
        print(f"Number of departments: {self.sales_df['Dept'].nunique()}")
        print(f"Total sales: ${self.sales_df['Weekly_Sales'].sum():,.2f}")
        print(f"Average weekly sales: ${self.sales_df['Weekly_Sales'].mean():,.2f}")
        
        # Aggregate sales by date
        self.aggregated_sales = DataPreprocessor.aggregate_by_date(self.sales_df)
        self.aggregated_sales = DataPreprocessor.create_time_features(self.aggregated_sales)
        
        print(f"\nAggregated time series length: {len(self.aggregated_sales)}")
    
    def plot_sales_trends(self, interactive=True):
        """Plot sales trends over time"""
        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.aggregated_sales['Date'],
                y=self.aggregated_sales['Weekly_Sales'],
                mode='lines',
                name='Weekly Sales',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title='Sales Trends Over Time',
                xaxis_title='Date',
                yaxis_title='Weekly Sales ($)',
                hovermode='x unified',
                template='plotly_white'
            )
            fig.show()
        else:
            VisualizationUtils.plot_sales_trends(self.aggregated_sales)
    
    def analyze_seasonality(self, interactive=True):
        """Analyze seasonal patterns"""
        if interactive:
            # Monthly seasonality
            monthly_avg = self.aggregated_sales.groupby('Month')['Weekly_Sales'].mean().reset_index()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Monthly Seasonality', 'Yearly Trends', 'Weekly Patterns', 'Quarterly Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Monthly seasonality
            fig.add_trace(
                go.Bar(x=monthly_avg['Month'], y=monthly_avg['Weekly_Sales'],
                      name='Monthly Avg', marker_color='lightblue'),
                row=1, col=1
            )
            
            # Yearly trends
            yearly_avg = self.aggregated_sales.groupby('Year')['Weekly_Sales'].mean().reset_index()
            fig.add_trace(
                go.Scatter(x=yearly_avg['Year'], y=yearly_avg['Weekly_Sales'],
                          mode='lines+markers', name='Yearly Avg', line=dict(color='red')),
                row=1, col=2
            )
            
            # Weekly patterns
            weekly_avg = self.aggregated_sales.groupby('DayOfWeek')['Weekly_Sales'].mean().reset_index()
            fig.add_trace(
                go.Bar(x=weekly_avg['DayOfWeek'], y=weekly_avg['Weekly_Sales'],
                      name='Weekly Avg', marker_color='lightgreen'),
                row=2, col=1
            )
            
            # Quarterly distribution
            quarterly_avg = self.aggregated_sales.groupby('Quarter')['Weekly_Sales'].mean().reset_index()
            fig.add_trace(
                go.Pie(labels=quarterly_avg['Quarter'], values=quarterly_avg['Weekly_Sales'],
                      name='Quarterly Distribution'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, title_text="Seasonality Analysis")
            fig.show()
        else:
            VisualizationUtils.plot_seasonality(self.aggregated_sales)
    
    def analyze_holiday_effects(self):
        """Analyze holiday effects on sales"""
        holiday_sales = self.sales_df[self.sales_df['IsHoliday'] == True]
        non_holiday_sales = self.sales_df[self.sales_df['IsHoliday'] == False]
        
        holiday_avg = holiday_sales['Weekly_Sales'].mean()
        non_holiday_avg = non_holiday_sales['Weekly_Sales'].mean()
        
        print(f"\n=== Holiday Effects Analysis ===")
        print(f"Holiday average sales: ${holiday_avg:,.2f}")
        print(f"Non-holiday average sales: ${non_holiday_avg:,.2f}")
        print(f"Holiday boost: {((holiday_avg/non_holiday_avg - 1) * 100):.1f}%")
        
        # Plot holiday effects
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=holiday_sales['Weekly_Sales'],
            name='Holiday Sales',
            marker_color='red',
            boxpoints='outliers'
        ))
        
        fig.add_trace(go.Box(
            y=non_holiday_sales['Weekly_Sales'],
            name='Non-Holiday Sales',
            marker_color='blue',
            boxpoints='outliers'
        ))
        
        fig.update_layout(
            title='Sales Distribution: Holiday vs Non-Holiday',
            yaxis_title='Weekly Sales ($)',
            template='plotly_white'
        )
        fig.show()
    
    def analyze_store_performance(self, top_n=10):
        """Analyze store performance"""
        store_performance = self.sales_df.groupby('Store').agg({
            'Weekly_Sales': ['mean', 'sum', 'count'],
            'Dept': 'nunique'
        }).round(2)
        
        store_performance.columns = ['Avg_Sales', 'Total_Sales', 'Records', 'Departments']
        store_performance = store_performance.sort_values('Total_Sales', ascending=False)
        
        print(f"\n=== Top {top_n} Performing Stores ===")
        print(store_performance.head(top_n))
        
        # Plot store performance
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top Stores by Total Sales', 'Store Sales Distribution',
                          'Average Sales by Store', 'Sales vs Departments'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Top stores
        top_stores = store_performance.head(top_n)
        fig.add_trace(
            go.Bar(x=top_stores.index, y=top_stores['Total_Sales'],
                  name='Total Sales', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Sales distribution
        fig.add_trace(
            go.Histogram(x=store_performance['Total_Sales'], nbinsx=20,
                        name='Sales Distribution', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Average sales
        fig.add_trace(
            go.Scatter(x=store_performance.index, y=store_performance['Avg_Sales'],
                      mode='lines+markers', name='Avg Sales', line=dict(color='red')),
            row=2, col=1
        )
        
        # Sales vs Departments
        fig.add_trace(
            go.Scatter(x=store_performance['Departments'], y=store_performance['Total_Sales'],
                      mode='markers', name='Sales vs Depts', marker=dict(size=8)),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Store Performance Analysis")
        fig.show()
        
        return store_performance
    
    def analyze_department_performance(self, top_n=10):
        """Analyze department performance"""
        dept_performance = self.sales_df.groupby('Dept').agg({
            'Weekly_Sales': ['mean', 'sum', 'count'],
            'Store': 'nunique'
        }).round(2)
        
        dept_performance.columns = ['Avg_Sales', 'Total_Sales', 'Records', 'Stores']
        dept_performance = dept_performance.sort_values('Total_Sales', ascending=False)
        
        print(f"\n=== Top {top_n} Performing Departments ===")
        print(dept_performance.head(top_n))
        
        # Plot department performance
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top Departments by Total Sales', 'Department Sales Distribution',
                          'Average Sales by Department', 'Sales vs Stores'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Top departments
        top_depts = dept_performance.head(top_n)
        fig.add_trace(
            go.Bar(x=top_depts.index, y=top_depts['Total_Sales'],
                  name='Total Sales', marker_color='lightcoral'),
            row=1, col=1
        )
        
        # Sales distribution
        fig.add_trace(
            go.Histogram(x=dept_performance['Total_Sales'], nbinsx=20,
                        name='Sales Distribution', marker_color='lightyellow'),
            row=1, col=2
        )
        
        # Average sales
        fig.add_trace(
            go.Scatter(x=dept_performance.index, y=dept_performance['Avg_Sales'],
                      mode='lines+markers', name='Avg Sales', line=dict(color='purple')),
            row=2, col=1
        )
        
        # Sales vs Stores
        fig.add_trace(
            go.Scatter(x=dept_performance['Stores'], y=dept_performance['Total_Sales'],
                      mode='markers', name='Sales vs Stores', marker=dict(size=8)),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Department Performance Analysis")
        fig.show()
        
        return dept_performance
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        # Create features for correlation analysis
        analysis_df = self.aggregated_sales.copy()
        
        # Add lag features
        analysis_df = DataPreprocessor.add_lag_features(analysis_df)
        
        # Add rolling features
        analysis_df = DataPreprocessor.add_rolling_features(analysis_df)
        
        # Select numeric columns
        numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if 'Weekly_Sales' in col or col in ['Year', 'Month', 'Week', 'DayOfWeek', 'Quarter']]
        
        # Plot correlation matrix
        corr_matrix = analysis_df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            xaxis_title='Features',
            yaxis_title='Features',
            height=600
        )
        fig.show()
        
        return corr_matrix
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*50)
        print("SALES FORECASTING - EXPLORATORY DATA ANALYSIS REPORT")
        print("="*50)
        
        # Basic statistics
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   • Total sales records: {len(self.sales_df):,}")
        print(f"   • Date range: {self.sales_df['Date'].min().strftime('%Y-%m-%d')} to {self.sales_df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"   • Number of stores: {self.sales_df['Store'].nunique()}")
        print(f"   • Number of departments: {self.sales_df['Dept'].nunique()}")
        print(f"   • Total sales volume: ${self.sales_df['Weekly_Sales'].sum():,.2f}")
        print(f"   • Average weekly sales: ${self.sales_df['Weekly_Sales'].mean():,.2f}")
        print(f"   • Sales standard deviation: ${self.sales_df['Weekly_Sales'].std():,.2f}")
        
        # Holiday analysis
        holiday_sales = self.sales_df[self.sales_df['IsHoliday'] == True]
        non_holiday_sales = self.sales_df[self.sales_df['IsHoliday'] == False]
        holiday_boost = ((holiday_sales['Weekly_Sales'].mean() / non_holiday_sales['Weekly_Sales'].mean() - 1) * 100)
        
        print(f"\n🎉 HOLIDAY EFFECTS:")
        print(f"   • Holiday average sales: ${holiday_sales['Weekly_Sales'].mean():,.2f}")
        print(f"   • Non-holiday average sales: ${non_holiday_sales['Weekly_Sales'].mean():,.2f}")
        print(f"   • Holiday sales boost: {holiday_boost:.1f}%")
        
        # Seasonal patterns
        monthly_avg = self.aggregated_sales.groupby('Month')['Weekly_Sales'].mean()
        best_month = monthly_avg.idxmax()
        worst_month = monthly_avg.idxmin()
        
        print(f"\n📅 SEASONAL PATTERNS:")
        print(f"   • Best performing month: {best_month} (${monthly_avg[best_month]:,.2f})")
        print(f"   • Worst performing month: {worst_month} (${monthly_avg[worst_month]:,.2f})")
        print(f"   • Seasonal variation: {((monthly_avg.max() / monthly_avg.min() - 1) * 100):.1f}%")
        
        # Store and department insights
        store_performance = self.sales_df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False)
        dept_performance = self.sales_df.groupby('Dept')['Weekly_Sales'].sum().sort_values(ascending=False)
        
        print(f"\n🏪 STORE & DEPARTMENT INSIGHTS:")
        print(f"   • Top performing store: Store {store_performance.index[0]} (${store_performance.iloc[0]:,.2f})")
        print(f"   • Top performing department: Dept {dept_performance.index[0]} (${dept_performance.iloc[0]:,.2f})")
        print(f"   • Store sales variation: {((store_performance.max() / store_performance.min() - 1) * 100):.1f}%")
        print(f"   • Department sales variation: {((dept_performance.max() / dept_performance.min() - 1) * 100):.1f}%")
        
        print("\n" + "="*50)
        print("END OF REPORT")
        print("="*50)

def main():
    """Main function to run EDA"""
    # Generate sample data if not exists
    try:
        analyzer = SalesAnalyzer('data/walmart_sales.csv', 'data/store_features.csv')
    except FileNotFoundError:
        print("Sample data not found. Generating sample data...")
        from src.data_generator import WalmartDataGenerator
        generator = WalmartDataGenerator()
        generator.create_sample_dataset()
        analyzer = SalesAnalyzer('data/walmart_sales.csv', 'data/store_features.csv')
    
    # Run comprehensive analysis
    analyzer.generate_summary_report()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_sales_trends()
    analyzer.analyze_seasonality()
    analyzer.analyze_holiday_effects()
    analyzer.analyze_store_performance()
    analyzer.analyze_department_performance()
    analyzer.correlation_analysis()
    
    print("\nEDA completed successfully!")

if __name__ == "__main__":
    main() 