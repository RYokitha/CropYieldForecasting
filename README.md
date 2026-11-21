# Climate-Based Agricultural Crop Yield Forecasting System

## Project Overview
This project uses Machine Learning to estimate agricultural crop yield based on historical data, including geographic area, crop type, year, annual rainfall, pesticide usage, and average temperature.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Random Forest Regressor
- Joblib
- Streamlit

## Dataset
- 28,242 records
- 101 geographic areas
- 10 crop types
- Historical data from 1990 to 2013

**Target variable:** `hg/ha_yield` (hectograms per hectare).

## Machine Learning Model
Algorithm: Random Forest Regressor

Chronological validation:
- Training period: 1990–2008
- Testing period: 2009–2013
- MAE: 10,336.35 hg/ha
- RMSE: 20,232.04 hg/ha
- R² Score: 95.47%

## Features
- Crop yield prediction
- Climate-related input analysis
- Multiple crop comparison
- Prediction history
- Data visualization
- CSV export

## Installation and Execution

1. Clone the repository:
   `git clone https://github.com/RYokitha/CropYieldForecasting.git`

2. Open the project folder:
   `cd CropYieldForecasting`

3. Create a virtual environment:
   `python -m venv venv`

4. Activate it on Windows:
   `venv\Scripts\activate`

5. Install dependencies:
   `pip install -r requirements.txt`

6. Run the application:
   `streamlit run app.py`

## Project Structure
- `dataset/` — Dataset files
- `notebooks/` — Data analysis notebook
- `models/` — Trained models and preprocessing files
- `src/` — Python source files
- `app.py` — Streamlit application
- `requirements.txt` — Project dependencies

## Limitations
- The dataset ends in 2013.
- It does not contain direct soil measurements.
- Predictions for years beyond 2013 are extrapolations and are not guaranteed actual yields.
- Results are estimates based on historical aggregated data.

## Author
Yokitha R.
