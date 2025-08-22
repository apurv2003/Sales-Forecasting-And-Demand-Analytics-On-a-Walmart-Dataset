#!/usr/bin/env python3
"""
Install Dependencies for Sales Forecasting Project
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("📦 Installing Sales Forecasting Project Dependencies")
    print("=" * 60)
    
    # Core dependencies
    core_packages = [
        "pandas==2.1.4",
        "numpy==1.24.3",
        "scikit-learn==1.3.2",
        "matplotlib==3.8.2",
        "seaborn==0.13.0"
    ]
    
    # Time series analysis
    timeseries_packages = [
        "statsmodels==0.14.0"
    ]
    
    # Visualization
    viz_packages = [
        "plotly==5.17.0",
        "dash==2.16.1",
        "dash-bootstrap-components==1.5.0"
    ]
    
    # Optional packages (may fail on some systems)
    optional_packages = [
        "prophet==1.1.4",
        "tensorflow==2.15.0",
        "keras==2.15.0"
    ]
    
    # Utilities
    util_packages = [
        "scipy==1.11.4",
        "joblib==1.3.2",
        "tqdm==4.66.1",
        "python-dateutil==2.8.2",
        "pytz==2023.3"
    ]
    
    print("\n📊 Installing core packages...")
    for package in core_packages:
        install_package(package)
    
    print("\n📈 Installing time series packages...")
    for package in timeseries_packages:
        install_package(package)
    
    print("\n📊 Installing visualization packages...")
    for package in viz_packages:
        install_package(package)
    
    print("\n🔧 Installing utility packages...")
    for package in util_packages:
        install_package(package)
    
    print("\n🤖 Installing optional packages (may fail)...")
    for package in optional_packages:
        install_package(package)
    
    print("\n" + "=" * 60)
    print("✅ Installation completed!")
    print("\n🚀 Next steps:")
    print("   1. Run 'python demo.py' to test the project")
    print("   2. Run 'python app.py' to start the dashboard")
    print("   3. Open http://localhost:8050 in your browser")

if __name__ == "__main__":
    main() 