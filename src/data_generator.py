"""
Data Generator for Walmart Sales Dataset
Creates synthetic data similar to the Walmart Weekly Sales dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class WalmartDataGenerator:
    def __init__(self):
        self.stores = list(range(1, 46))  # 45 stores
        self.departments = list(range(1, 100))  # 99 departments
        self.holidays = [
            'Super Bowl', 'Labor Day', 'Thanksgiving', 'Christmas'
        ]
        
    def generate_date_range(self, start_date='2010-02-05', end_date='2012-11-01'):
        """Generate weekly date range"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=7)
        return dates
    
    def generate_holiday_flags(self, dates):
        """Generate holiday flags based on known Walmart holidays"""
        holiday_flags = []
        for date in dates:
            dt = datetime.strptime(date, '%Y-%m-%d')
            month = dt.month
            day = dt.day
            
            # Simple holiday logic
            if month == 2 and day in [12, 19, 26]:  # Super Bowl (approximate)
                holiday_flags.append(True)
            elif month == 9 and day in [10, 17, 24]:  # Labor Day (approximate)
                holiday_flags.append(True)
            elif month == 11 and day in [26]:  # Thanksgiving
                holiday_flags.append(True)
            elif month == 12 and day in [23, 30]:  # Christmas
                holiday_flags.append(True)
            else:
                holiday_flags.append(False)
        
        return holiday_flags
    
    def generate_sales_data(self, num_stores=45, num_depts=99):
        """Generate synthetic sales data"""
        dates = self.generate_date_range()
        holiday_flags = self.generate_holiday_flags(dates)
        
        data = []
        
        for store in range(1, num_stores + 1):
            for dept in range(1, num_depts + 1):
                # Skip some department-store combinations (realistic)
                if random.random() < 0.3:
                    continue
                    
                for i, date in enumerate(dates):
                    # Base sales with store and department effects
                    base_sales = 5000 + (store * 100) + (dept * 50)
                    
                    # Add seasonal trend
                    dt = datetime.strptime(date, '%Y-%m-%d')
                    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * dt.timetuple().tm_yday / 365)
                    
                    # Add holiday effect
                    holiday_boost = 1.5 if holiday_flags[i] else 1.0
                    
                    # Add random noise
                    noise = np.random.normal(0, 0.1)
                    
                    # Calculate final sales
                    sales = base_sales * seasonal_factor * holiday_boost * (1 + noise)
                    sales = max(0, sales)  # Ensure non-negative
                    
                    data.append({
                        'Store': store,
                        'Dept': dept,
                        'Date': date,
                        'Weekly_Sales': round(sales, 2),
                        'IsHoliday': holiday_flags[i]
                    })
        
        return pd.DataFrame(data)
    
    def generate_store_features(self):
        """Generate store features data"""
        store_data = []
        for store in self.stores:
            store_data.append({
                'Store': store,
                'Type': random.choice(['A', 'B', 'C']),
                'Size': random.randint(40000, 200000),
                'Temperature': random.uniform(30, 80),
                'Fuel_Price': random.uniform(2.5, 4.5),
                'MarkDown1': random.uniform(0, 5000),
                'MarkDown2': random.uniform(0, 5000),
                'MarkDown3': random.uniform(0, 5000),
                'MarkDown4': random.uniform(0, 5000),
                'MarkDown5': random.uniform(0, 5000),
                'CPI': random.uniform(120, 200),
                'Unemployment': random.uniform(5, 15)
            })
        return pd.DataFrame(store_data)
    
    def create_sample_dataset(self):
        """Create complete sample dataset"""
        print("Generating sales data...")
        sales_df = self.generate_sales_data()
        
        print("Generating store features...")
        store_df = self.generate_store_features()
        
        # Save datasets
        sales_df.to_csv('data/walmart_sales.csv', index=False)
        store_df.to_csv('data/store_features.csv', index=False)
        
        print(f"Generated {len(sales_df)} sales records")
        print(f"Generated {len(store_df)} store records")
        
        return sales_df, store_df

if __name__ == "__main__":
    generator = WalmartDataGenerator()
    sales_data, store_data = generator.create_sample_dataset()
    print("Sample datasets created successfully!") 