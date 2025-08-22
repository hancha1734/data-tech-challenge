# Generate: Data Branch Tech Challenge

My clean csv: https://docs.google.com/spreadsheets/d/1srUVsIoycalQRkL-BpmXXa-NVZir351-4DkkT0waVBs/edit?usp=sharing

My process:
I started by cleaning up the flight data so it was easy to work with. That meant making sure dates were in the right format, handling any missing values, creating complete routes, and deleting columns I didn't want. I looked at the data to see which routes were busiest, which were slow, and if there were any seasonal patterns.

To understand the trends quickly, I made bar charts showing passengers, freight, and mail for the top 10 and bottom 10 countries by passenger numbers. I used grouped bars and a second axis so you could compare everything in one graph. It gave a clear picture of where AeroConnect might make more profit.

For building the model, I picked Sydney-Auckland because it’s the busiest route. I created features like past months’ passenger numbers (lags) and encoded month and year to help the model spot trends over time. Then I built a Random Forest Regression model with time-series cross-validation to get the best predictions.

I measured how well the model did using metrics like MSE, RMSE, MAE, MAPE, and R². The model was really good for Sydney-Auckland: it had an R² of 0.9077, meaning it explains over 90% of the variation in passenger numbers, and an RMSE of 2,832 passengers, which is only about 5% of the average monthly passengers.

Using the model, I predicted passenger numbers for the next 12 months. I plotted the forecasts alongside historical data so it’s easy to see how the model is doing and what to expect next.