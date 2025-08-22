import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import numpy as np
from model import random_forest_forecast


# Data Loading Functions ---------------------------------------------------------------------

def load_data(filename: str) -> pd.DataFrame:
    """Load CSV file into a pandas DataFrame and show basic info."""
    df = pd.read_csv(filename)
    print("Data loaded. Info:")
    print(df.info())  # Print summary of DataFrame: columns, types, non-null counts
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to consistent snake_case names for readability and coding standards."""
    df.rename(columns={
        'Passengers_In': 'passengers_in',
        'Passengers_Out': 'passengers_out',
        'Passengers_Total': 'passengers_total',
        'Freight_In_(tonnes)': 'freight_in_tonnes',
        'Freight_Out_(tonnes)': 'freight_out_tonnes',
        'Freight_Total_(tonnes)': 'freight_total_tonnes',
        'Mail_In_(tonnes)': 'mail_in_tonnes',
        'Mail_Out_(tonnes)': 'mail_out_tonnes',
        'Mail_Total_(tonnes)': 'mail_total_tonnes',
        'AustralianPort': 'australian_port',
        'ForeignPort': 'foreign_port',
        'Country': 'country',
        'Month': 'month',
        'Year': 'year',
        'Month_num': 'month_num'
    }, inplace=True)
    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and standardize capitalization for text columns."""
    for col in ['australian_port', 'foreign_port', 'country']:
        df[col] = df[col].astype(str).str.strip().str.title()  # Capitalize first letters
    return df


# Data Transformation ---------------------------------------------------------------------

def create_additional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create Route, MonthYear, and MonthYear as datetime for plotting."""
    # Combine origin and destination into a single route column
    df['route'] = df['australian_port'] + "-" + df['foreign_port']

    # Create a proper datetime column for chronological plotting
    # Assumes day = 1 for all dates
    df['month_year_dt'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month_num'].astype(str) + '-01')

    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop original columns no longer needed after totals and route creation."""
    df = df.drop(columns=[
        "month", "australian_port", "foreign_port",
        "passengers_in", "freight_in_tonnes", "mail_in_tonnes",
        "passengers_out", "freight_out_tonnes", "mail_out_tonnes",
        "year", "month_num"
    ])
    return df


def save_cleaned_data(df: pd.DataFrame, filename: str):
    """Save the cleaned DataFrame to a new CSV file."""
    df.to_csv(filename, index=False)
    print(f"Cleaned data saved to {filename}")


# Analysis Functions ---------------------------------------------------------------------

def top_bottom_routes(df: pd.DataFrame, top_n: int = 10):
    """
    Return the top N most and least active routes based on total passengers.
    Excludes routes with 0 passengers.
    """
    # Aggregate passenger totals by route
    route_traffic = df.groupby('route')['passengers_total'].sum().reset_index()

    # Remove routes with no passenger activity
    active_routes = route_traffic[route_traffic['passengers_total'] > 0]

    # Sort descending for top routes
    sorted_routes = active_routes.sort_values(by='passengers_total', ascending=False)

    # Select top N busiest routes
    top_routes = sorted_routes.head(top_n)

    # Select bottom N least active routes and reverse order so least is first
    bottom_routes = sorted_routes.tail(top_n).sort_values(by='passengers_total', ascending=True)

    print(f"Top {top_n} most active routes:")
    print(top_routes)
    print("\n")
    print(f"Top {top_n} least active routes: (excluding trips with 0 passengers)")
    print(bottom_routes)

    return top_routes, bottom_routes


def plot_passenger_trends(df: pd.DataFrame, route: str = None):
    """
    Plot passenger trends over time.
    If route is None, plot total passengers across all routes.
    """
    if route:
        data = df[df['route'] == route]
        title = f"Passenger Trend for {route}"
    else:
        # Aggregate passengers across all routes per month-year
        data = df.groupby('month_year_dt')['passengers_total'].sum().reset_index()
        title = "Total Passengers Across All Routes"

    plt.figure(figsize=(12, 6))
    plt.plot(data['month_year_dt'], data['passengers_total'], marker='o')
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel("Month-Year")
    plt.ylabel("Total Passengers")
    plt.tight_layout()
    plt.show()


def plot_geographical_patterns(df: pd.DataFrame, top_n: int = 10):
    """
    Plot top N busiest destinations by country to show geographical trends.
    Adjust x-axis ticks for readability.
    """
    # Aggregate total passengers by country and select top N
    country_traffic = df.groupby('country')['passengers_total'].sum().sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    country_traffic.plot(kind='bar')
    plt.title(f"Top {top_n} Busiest Destinations by Country")
    plt.ylabel("Total Passengers")

    # Rotate ticks and adjust font size for readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    plt.gca().get_yaxis().set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

    plt.tight_layout()
    plt.show()

def plot_seasonal_heatmap(df: pd.DataFrame):
    """
    Plot a heatmap of total passengers: Years vs Months.
    Highlights seasonal and yearly trends.
    """
    # Pivot the data: rows = year, columns = month, values = total passengers
    pivot = df.pivot_table(
        index=df['month_year_dt'].dt.year,        # Year on y-axis
        columns=df['month_year_dt'].dt.month,     # Month on x-axis
        values='passengers_total',                # Total passengers
        aggfunc='sum'
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt="g", cmap="YlGnBu")  # Color-coded values
    plt.title("Yearly and Monthly Passenger Traffic Heatmap")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.show()

def top_routes_by_country(df: pd.DataFrame, country: str, top_n: int = 5):
        """Show top N routes for a specific country to identify key city pairs."""
        data = df[df['country'] == country].groupby('route')['passengers_total'].sum().sort_values(
            ascending=False).head(top_n)
        print(f"Top {top_n} routes for {country}:")
        print(data)
        return data

def plot_top_routes_by_country(df: pd.DataFrame, country: str, top_n: int = 5):
    """
    Plot a bar chart of the top N busiest routes to a specific country.
    """
    # Get top routes using your existing function
    top_routes = top_routes_by_country(df, country, top_n)

    # Plotting
    plt.figure(figsize=(10, 5))
    top_routes.plot(kind='bar')
    plt.title(f"Top {top_n} Routes to {country}")
    plt.xlabel("Route")
    plt.ylabel("Total Passengers")
    plt.xticks(rotation=45)
    plt.gca().get_yaxis().set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    plt.tight_layout()
    plt.show()


def plot_countries_comparison(df: pd.DataFrame, n: int = 10, top: bool = True):
    """
    Plot a grouped bar chart comparing total passengers, freight, and mail
    for either top or bottom N countries by passenger traffic using a secondary y-axis.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.
        n (int): Number of countries to display.
        top (bool): If True, plot top N countries; if False, plot bottom N countries.
    """
    # Aggregate totals by country
    country_totals = df.groupby('country')[['passengers_total', 'freight_total_tonnes', 'mail_total_tonnes']].sum()

    # Sort by passengers
    sorted_countries = country_totals.sort_values(by='passengers_total', ascending=not top).head(n)

    countries = sorted_countries.index
    x = np.arange(len(countries))  # label locations
    width = 0.25  # width of each bar

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Bar for passengers (primary y-axis)
    bars_passengers = ax1.bar(x - width, sorted_countries['passengers_total'], width, color='skyblue',
                              label='Passengers')
    ax1.set_ylabel('Total Passengers', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    # Secondary y-axis for freight and mail
    ax2 = ax1.twinx()
    bars_freight = ax2.bar(x, sorted_countries['freight_total_tonnes'], width, color='green', label='Freight (tonnes)')
    bars_mail = ax2.bar(x + width, sorted_countries['mail_total_tonnes'], width, color='orange', label='Mail (tonnes)')
    ax2.set_ylabel('Freight / Mail (tonnes)')
    ax2.tick_params(axis='y')

    # X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(countries, rotation=45, ha='right')

    # Legends
    ax1.legend([bars_passengers], ['Passengers'], loc='upper left')
    ax2.legend([bars_freight, bars_mail], ['Freight', 'Mail'], loc='upper right')

    kind = "Top" if top else "Bottom"
    plt.title(f'{kind} {n} Countries by Total Passengers: Passengers, Freight, and Mail')
    plt.tight_layout()
    plt.show()


# Main Execution ---------------------------------------------------------------------

def main():
    # Load the original CSV file
    df = load_data("TechChallenge_Data.csv")

    # Standardize column names and clean text
    df = rename_columns(df)
    df = clean_text_columns(df)

    # Create additional columns (route, month-year, totals)
    df = create_additional_columns(df)

    # Drop columns that are no longer needed
    df = drop_unnecessary_columns(df)

    # Save cleaned dataset
    save_cleaned_data(df, "TechChallenge_Data_cleaned.csv")

    # Identify top and bottom routes
    route_traffic = top_bottom_routes(df)


    rf_model, forecast = random_forest_forecast(df, forecast_months=12)

    # Plot passenger trends for all routes
    plot_passenger_trends(df)

    # Plot passenger trends for a specific route
    plot_passenger_trends(df, route="Sydney-Auckland")

    # Plot geographical trends
    plot_geographical_patterns(df)

    # Plot seasonal and yearly trends with a heatmap
    plot_seasonal_heatmap(df)

    plot_top_routes_by_country(df, country="New Zealand", top_n=5)

    # Top 10 countries
    plot_countries_comparison(df, n=10, top=True)

    # Bottom 10 countries
    plot_countries_comparison(df, n=10, top=False)



if __name__ == "__main__":
    main()
