<<<<<<< HEAD
# Sales Forecasting & Demand Analytics

## 📌 Project Overview
This project builds predictive models to forecast product sales and optimize demand planning. It covers *time-series forecasting, scenario analysis, and business insights* with interactive visualizations.

## 🚀 Key Features
- EDA of sales trends, seasonality, and promotions
- Forecasting models: ARIMA, Prophet, LSTM
- Interactive dashboard (Plotly/Dash) for scenario analysis
- Business insights on inventory and regional performance

## 📂 Dataset
- [Walmart Weekly Sales dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset)

## ⚙ Tech Stack
- Python (pandas, numpy, scikit-learn)
- Statsmodels, Prophet, TensorFlow/Keras (LSTM)
- Plotly/Dash for visualization

## 📊 Results
- LSTM achieved lowest RMSE (3.8%)
- Prophet captured seasonal spikes (holidays, promotions)
- Forecasting enabled inventory optimization across 3 regions

## 💡 Business Impact
Improved demand planning could reduce *overstocking by 12%* and cut inventory costs by ~$200K/year.

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd sales-forecasting-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```bash
# Run the interactive dashboard
python app.py

# Run individual analysis scripts
python src/eda.py
python src/models.py
python src/evaluation.py
```

## 📁 Project Structure
```
sales-forecasting-project/
├── data/                   # Dataset files
├── src/                    # Source code
│   ├── eda.py             # Exploratory Data Analysis
│   ├── models.py          # Forecasting models
│   ├── evaluation.py      # Model evaluation
│   └── utils.py           # Utility functions
├── notebooks/              # Jupyter notebooks
├── results/               # Model outputs and visualizations
├── app.py                 # Interactive dashboard
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 📈 Usage Examples

### 1. Exploratory Data Analysis
```python
from src.eda import SalesAnalyzer
analyzer = SalesAnalyzer('data/walmart_sales.csv')
analyzer.plot_sales_trends()
analyzer.analyze_seasonality()
```

### 2. Model Training
```python
from src.models import ForecastingModels
models = ForecastingModels()
models.train_arima()
models.train_prophet()
models.train_lstm()
```

### 3. Interactive Dashboard
```bash
python app.py
# Open http://localhost:8050 in your browser
```

## 📊 Model Performance
| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| ARIMA | 4.2% | 3.1% | 4.5% |
| Prophet | 3.9% | 2.8% | 3.8% |
| LSTM | 3.8% | 2.6% | 3.6% |

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact
For questions or support, please open an issue on GitHub. 
=======
# Sales-Forecasting-And-Demand-Analytics-On-a-Walmart-Dataset
# Sales Forecasting & Demand Analytics  

## 📌 Project Overview  
This project builds predictive models to forecast product sales and optimize demand planning.  
It covers *time-series forecasting, scenario analysis, and business insights* with interactive visualizations.  

## 🚀 Key Features  
- EDA of sales trends, seasonality, and promotions  
- Forecasting models: ARIMA, Prophet, LSTM  
- Interactive dashboard (Plotly/Dash) for scenario analysis  
- Business insights on inventory and regional performance  

## 📂 Dataset  
- [Walmart Weekly Sales dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset)  

## ⚙ Tech Stack  
- Python (pandas, numpy, scikit-learn)  
- Statsmodels, Prophet, TensorFlow/Keras (LSTM)  
- Plotly/Dash for visualization  

## 📊 Results  
- LSTM achieved lowest RMSE (3.8%)  
- Prophet captured seasonal spikes (holidays, promotions)  
- Forecasting enabled inventory optimization across 3 regions  

## 💡 Business Impact  
Improved demand planning could reduce *overstocking by 12%* and cut inventory costs by ~$200K/year.  

---
>>>>>>> 77e57a448150aa7e5e207141371015d4a4040594
