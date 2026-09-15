\# Supply Chain Demand Forecasting



\## Overview



Supply Chain Demand Forecasting is a machine learning application that analyzes historical demand and forecasts future product demand to support inventory and supply planning.



\## Features



\- CSV dataset upload

\- ZIP file upload containing CSV datasets

\- Automatic date column detection

\- Automatic demand column detection

\- Historical demand analysis

\- Random Forest regression

\- Lag-based demand features

\- Rolling demand analysis

\- Forecast visualization

\- Future demand forecasting

\- MAE and RMSE evaluation

\- Supply chain insights

\- Forecast CSV download

\- Interactive Streamlit dashboard



\## Technologies Used



\- Python

\- Streamlit

\- Pandas

\- NumPy

\- Scikit-learn

\- Random Forest Regression



\## Machine Learning Approach



The system uses historical demand features including:



\- Previous-day demand

\- Previous-week demand

\- Seven-day rolling average

\- Day of the week



A Random Forest Regression model learns relationships between these features and demand.



\## Input



The application accepts:



\- CSV files

\- ZIP files containing CSV datasets



The dataset should contain a date column and a numerical demand, sales, quantity, units, or orders column.



\## Output



The application provides:



\- Historical demand visualization

\- Forecast demand

\- Future demand predictions

\- MAE

\- RMSE

\- Supply chain recommendations

\- Downloadable forecast results



\## How to Run



Install dependencies:



```bash

pip install -r requirements.txt

