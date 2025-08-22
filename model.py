import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def prepare_sydney_auckland_data(df: pd.DataFrame, lags=3):
    """
    Prepares features and target for Sydney-Auckland passenger traffic.
    Adds lag features and month/year encoding.
    """
    # Filter Sydney-Auckland
    df_route = df[df['route'] == 'Sydney-Auckland'].sort_values('month_year_dt').copy()

    # Create lag features
    for lag in range(1, lags+1):
        df_route[f'lag_{lag}'] = df_route['passengers_total'].shift(lag)

    # Encode month and year as numerical/categorical
    df_route['month'] = df_route['month_year_dt'].dt.month
    df_route['year'] = df_route['month_year_dt'].dt.year

    # Drop rows with NaN values due to lagging
    df_route = df_route.dropna()

    # Features and target
    features = [f'lag_{lag}' for lag in range(1, lags+1)] + ['month', 'year']
    X = df_route[features]
    y = df_route['passengers_total']

    return X, y, df_route

def random_forest_forecast(df: pd.DataFrame, forecast_months=12):
    """
    Trains Random Forest on Sydney-Auckland data, evaluates performance,
    and forecasts next 6-12 months.
    """
    X, y, df_route = prepare_sydney_auckland_data(df)

    # Time series split (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)

    # Random Forest with hyperparameter grid search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=tscv, scoring='r2')
    grid.fit(X, y)

    best_model = grid.best_estimator_

    # Predictions on training data
    y_pred = best_model.predict(X)

    # Evaluate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - y_pred))
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    mean_y = np.mean(y)
    r2 = r2_score(y, y_pred)

    print("Random Forest Performance for Sydney-Auckland:")
    print("Best Parameters:", grid.best_params_)
    print(f"Mean Squared Error (MSE): {mse:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} passengers")
    print(f"Mean Absolute Error (MAE): {mae:,.2f} passengers")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Mean monthly passengers: {mean_y:,.0f}")
    print(f"RMSE as % of mean passengers: {rmse/mean_y*100:.2f}%")
    print(f"R^2 Score: {r2:.4f}")

    # Forecast next months
    last_row = df_route.iloc[-1]
    forecast_values = []
    lag_values = [last_row[f'lag_{i}'] for i in range(1, 4)]

    for i in range(forecast_months):
        month = (last_row['month'] + i + 1 - 1) % 12 + 1
        year = last_row['year'] + (last_row['month'] + i) // 12
        features_df = pd.DataFrame([lag_values + [month, year]], columns=X.columns)
        forecast = best_model.predict(features_df)[0]
        forecast_values.append(forecast)
        lag_values = lag_values[1:] + [forecast]

    # Plot actual vs forecast
    plt.figure(figsize=(12, 6))
    plt.plot(df_route['month_year_dt'], df_route['passengers_total'], marker='o', label='Actual')
    forecast_dates = pd.date_range(df_route['month_year_dt'].iloc[-1] + pd.DateOffset(months=1),
                                   periods=forecast_months, freq='MS')
    plt.plot(forecast_dates, forecast_values, marker='x', linestyle='--', label='Forecast')
    plt.title('Sydney-Auckland Passengers: Actual vs Forecast')
    plt.xlabel('Date')
    plt.ylabel('Passengers')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return best_model, forecast_values

