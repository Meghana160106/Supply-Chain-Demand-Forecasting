\# Machine Learning Model Information



\## Model Used



The project uses a Random Forest Regression model for demand forecasting.



\## Input Features



The model uses the following features:



\- Previous-day demand

\- Previous-week demand

\- Seven-day rolling average

\- Day of the week



\## Why Random Forest?



Random Forest Regression is suitable for this prototype because it can capture nonlinear relationships between historical demand patterns and future demand.



It also works well with multiple numerical features and does not require complex mathematical assumptions about the relationship between variables.



\## Training Process



1\. Historical demand data is loaded.

2\. Date and demand columns are identified.

3\. Data is cleaned and sorted chronologically.

4\. Lag features are created.

5\. A seven-day rolling average is calculated.

6\. The dataset is divided into training and testing portions.

7\. The Random Forest model is trained.

8\. Predictions are generated for the test data.

9\. MAE and RMSE are calculated.

10\. The trained model is used for future demand forecasting.



\## Evaluation Metrics



\### Mean Absolute Error (MAE)



MAE measures the average absolute difference between actual demand and predicted demand.



\### Root Mean Squared Error (RMSE)



RMSE measures prediction error while giving greater weight to larger errors.



Lower MAE and RMSE values indicate better forecasting performance.



\## Forecasting Strategy



The model uses recent historical predictions recursively to generate forecasts for future days.



The application supports forecasting horizons from 7 to 30 days.



\## Future Improvements



The model can be improved using:



\- LSTM networks

\- XGBoost

\- Seasonal forecasting

\- Product-specific models

\- Holiday and promotion features

\- Supplier lead-time information

\- Real-time inventory data

