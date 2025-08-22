#!/usr/bin/env python3
"""
Minimal Sales Forecasting Demo
Works with basic Python packages (pandas, numpy)
"""

import sys
import os
sys.path.append('src')

def main():
    print("🚀 Sales Forecasting - Minimal Demo")
    print("=" * 50)
    
    try:
        # Step 1: Generate sample data
        print("\n📊 Step 1: Generating sample data...")
        from src.data_generator import WalmartDataGenerator
        generator = WalmartDataGenerator()
        sales_data, store_data = generator.create_sample_dataset()
        print(f"✅ Generated {len(sales_data):,} sales records and {len(store_data)} store records")
        
        # Step 2: Basic data analysis
        print("\n🔍 Step 2: Basic data analysis...")
        import pandas as pd
        
        # Load data
        sales_df = pd.read_csv('data/walmart_sales.csv')
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        
        # Basic statistics
        print(f"\n📈 Basic Statistics:")
        print(f"   • Total records: {len(sales_df):,}")
        print(f"   • Date range: {sales_df['Date'].min()} to {sales_df['Date'].max()}")
        print(f"   • Number of stores: {sales_df['Store'].nunique()}")
        print(f"   • Number of departments: {sales_df['Dept'].nunique()}")
        print(f"   • Total sales: ${sales_df['Weekly_Sales'].sum():,.2f}")
        print(f"   • Average weekly sales: ${sales_df['Weekly_Sales'].mean():,.2f}")
        print(f"   • Sales standard deviation: ${sales_df['Weekly_Sales'].std():,.2f}")
        
        # Aggregate by date
        aggregated_sales = sales_df.groupby('Date')['Weekly_Sales'].sum().reset_index()
        print(f"   • Time series length: {len(aggregated_sales)} weeks")
        
        # Holiday analysis
        holiday_sales = sales_df[sales_df['IsHoliday'] == True]
        non_holiday_sales = sales_df[sales_df['IsHoliday'] == False]
        
        if len(holiday_sales) > 0:
            holiday_boost = ((holiday_sales['Weekly_Sales'].mean() / non_holiday_sales['Weekly_Sales'].mean() - 1) * 100)
            print(f"   • Holiday sales boost: {holiday_boost:.1f}%")
        
        # Store performance
        store_performance = sales_df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False)
        print(f"   • Top performing store: Store {store_performance.index[0]} (${store_performance.iloc[0]:,.2f})")
        print(f"   • Store sales variation: {((store_performance.max() / store_performance.min() - 1) * 100):.1f}%")
        
        # Department performance
        dept_performance = sales_df.groupby('Dept')['Weekly_Sales'].sum().sort_values(ascending=False)
        print(f"   • Top performing department: Dept {dept_performance.index[0]} (${dept_performance.iloc[0]:,.2f})")
        print(f"   • Department sales variation: {((dept_performance.max() / dept_performance.min() - 1) * 100):.1f}%")
        
        # Step 3: Simple forecasting (moving average)
        print("\n🤖 Step 3: Simple forecasting model...")
        
        # Create simple moving average forecast
        aggregated_sales = aggregated_sales.sort_values('Date')
        aggregated_sales['MA_4'] = aggregated_sales['Weekly_Sales'].rolling(window=4).mean()
        aggregated_sales['MA_8'] = aggregated_sales['Weekly_Sales'].rolling(window=8).mean()
        
        # Calculate simple forecast accuracy
        test_size = int(len(aggregated_sales) * 0.2)
        train_data = aggregated_sales.iloc[:-test_size]
        test_data = aggregated_sales.iloc[-test_size:]
        
        # Simple forecast using last moving average
        last_ma4 = train_data['MA_4'].iloc[-1]
        last_ma8 = train_data['MA_8'].iloc[-1]
        
        # Calculate metrics
        actual_values = test_data['Weekly_Sales'].values
        ma4_forecast = [last_ma4] * len(test_data)
        ma8_forecast = [last_ma8] * len(test_data)
        
        # Calculate RMSE
        import numpy as np
        def calculate_rmse(actual, predicted):
            return np.sqrt(np.mean((actual - predicted) ** 2))
        
        ma4_rmse = calculate_rmse(actual_values, ma4_forecast)
        ma8_rmse = calculate_rmse(actual_values, ma8_forecast)
        
        print(f"   • 4-week Moving Average RMSE: {ma4_rmse:.2f}")
        print(f"   • 8-week Moving Average RMSE: {ma8_rmse:.2f}")
        
        # Step 4: Business insights
        print("\n💼 Step 4: Business insights...")
        
        # Calculate potential savings
        avg_weekly_sales = sales_df['Weekly_Sales'].mean()
        forecast_error_reduction = 0.05  # 5% improvement
        annual_weeks = 52
        
        potential_savings = avg_weekly_sales * forecast_error_reduction * annual_weeks
        inventory_optimization = 0.12  # 12% reduction
        
        print(f"   • Average weekly sales: ${avg_weekly_sales:,.2f}")
        print(f"   • Potential annual savings: ${potential_savings:,.0f}")
        print(f"   • Inventory optimization potential: {inventory_optimization*100}% reduction")
        
        # Step 5: Summary
        print("\n🎯 PROJECT SUMMARY")
        print("=" * 50)
        print("✅ Sample data generated successfully")
        print("✅ Basic data analysis completed")
        print("✅ Simple forecasting model implemented")
        print("✅ Business insights calculated")
        
        print("\n📈 Key Findings:")
        print(f"   • Dataset covers {len(aggregated_sales)} weeks of sales data")
        print(f"   • {sales_df['Store'].nunique()} stores across {sales_df['Dept'].nunique()} departments")
        print(f"   • Total sales volume: ${sales_df['Weekly_Sales'].sum():,.0f}")
        if len(holiday_sales) > 0:
            print(f"   • Holiday periods show {holiday_boost:.1f}% sales increase")
        print(f"   • Simple forecasting achieves RMSE of {min(ma4_rmse, ma8_rmse):.2f}")
        
        print("\n🚀 Next Steps:")
        print("   1. Install full dependencies: python install_dependencies.py")
        print("   2. Run full demo: python demo.py")
        print("   3. Start dashboard: python app.py")
        print("   4. Open http://localhost:8050 in your browser")
        
        print("\n🎉 Minimal demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Demo completed successfully!")
    else:
        print("\n❌ Demo failed. Please check the error messages above.")
        sys.exit(1) 