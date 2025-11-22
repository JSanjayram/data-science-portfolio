import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Generate temperature time series data
def create_temperature_data():
    # Generate 10 years of daily temperature data
    start_date = datetime(2014, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)
    n_days = len(date_range)
    
    # Create realistic temperature patterns
    # Base temperature with seasonal variation
    day_of_year = date_range.dayofyear
    seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    # Add yearly warming trend (climate change simulation)
    years = date_range.year
    yearly_trend = (years - 2014) * 0.1
    
    # Add random daily variation
    daily_variation = np.random.normal(0, 3, n_days)
    
    # Combine all components
    temperature = seasonal_temp + yearly_trend + daily_variation
    
    # Create multiple cities with different base temperatures
    cities = ['New York', 'London', 'Tokyo', 'Sydney', 'Mumbai']
    city_offsets = [0, -3, 2, 5, 8]  # Temperature offsets for different cities
    
    data = []
    for city, offset in zip(cities, city_offsets):
        city_temp = temperature + offset + np.random.normal(0, 1, n_days)
        for i, date in enumerate(date_range):
            data.append({
                'date': date,
                'city': city,
                'temperature': city_temp[i],
                'year': date.year,
                'month': date.month,
                'day_of_year': date.dayofyear,
                'season': get_season(date.month)
            })
    
    return pd.DataFrame(data)

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

# Create comprehensive time series visualizations
def create_timeseries_plots(df):
    plt.figure(figsize=(20, 15))
    
    # 1. Overall temperature trends for all cities
    plt.subplot(3, 4, 1)
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        plt.plot(city_data['date'], city_data['temperature'], alpha=0.7, label=city)
    plt.title('Temperature Trends Across Years (All Cities)')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # 2. Yearly average temperatures
    plt.subplot(3, 4, 2)
    yearly_avg = df.groupby(['year', 'city'])['temperature'].mean().reset_index()
    for city in df['city'].unique():
        city_yearly = yearly_avg[yearly_avg['city'] == city]
        plt.plot(city_yearly['year'], city_yearly['temperature'], marker='o', label=city)
    plt.title('Yearly Average Temperature Trends')
    plt.xlabel('Year')
    plt.ylabel('Average Temperature (°C)')
    plt.legend()
    
    # 3. Seasonal patterns
    plt.subplot(3, 4, 3)
    seasonal_avg = df.groupby(['month', 'city'])['temperature'].mean().reset_index()
    for city in df['city'].unique():
        city_seasonal = seasonal_avg[seasonal_avg['city'] == city]
        plt.plot(city_seasonal['month'], city_seasonal['temperature'], marker='o', label=city)
    plt.title('Seasonal Temperature Patterns')
    plt.xlabel('Month')
    plt.ylabel('Average Temperature (°C)')
    plt.legend()
    
    # 4. Temperature distribution by city
    plt.subplot(3, 4, 4)
    df.boxplot(column='temperature', by='city', ax=plt.gca())
    plt.title('Temperature Distribution by City')
    plt.suptitle('')
    plt.xticks(rotation=45)
    
    # 5. Heatmap of monthly temperatures
    plt.subplot(3, 4, 5)
    monthly_pivot = df.groupby(['year', 'month'])['temperature'].mean().reset_index()
    heatmap_data = monthly_pivot.pivot(index='year', columns='month', values='temperature')
    sns.heatmap(heatmap_data, cmap='RdYlBu_r', cbar_kws={'label': 'Temperature (°C)'})
    plt.title('Monthly Temperature Heatmap')
    
    # 6. Temperature anomalies
    plt.subplot(3, 4, 6)
    overall_mean = df['temperature'].mean()
    df['anomaly'] = df['temperature'] - overall_mean
    yearly_anomaly = df.groupby('year')['anomaly'].mean()
    plt.bar(yearly_anomaly.index, yearly_anomaly.values, 
            color=['blue' if x < 0 else 'red' for x in yearly_anomaly.values])
    plt.title('Yearly Temperature Anomalies')
    plt.xlabel('Year')
    plt.ylabel('Temperature Anomaly (°C)')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 7. Rolling average trends
    plt.subplot(3, 4, 7)
    ny_data = df[df['city'] == 'New York'].copy()
    ny_data = ny_data.sort_values('date')
    ny_data['rolling_30'] = ny_data['temperature'].rolling(window=30).mean()
    ny_data['rolling_365'] = ny_data['temperature'].rolling(window=365).mean()
    
    plt.plot(ny_data['date'], ny_data['temperature'], alpha=0.3, label='Daily')
    plt.plot(ny_data['date'], ny_data['rolling_30'], label='30-day average')
    plt.plot(ny_data['date'], ny_data['rolling_365'], label='365-day average')
    plt.title('Temperature Trends - New York (Rolling Averages)')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # 8. Seasonal comparison across years
    plt.subplot(3, 4, 8)
    seasonal_yearly = df.groupby(['year', 'season'])['temperature'].mean().reset_index()
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    for season in seasons:
        season_data = seasonal_yearly[seasonal_yearly['season'] == season]
        plt.plot(season_data['year'], season_data['temperature'], marker='o', label=season)
    plt.title('Seasonal Temperature Trends Over Years')
    plt.xlabel('Year')
    plt.ylabel('Average Temperature (°C)')
    plt.legend()
    
    # 9. Temperature range by year
    plt.subplot(3, 4, 9)
    yearly_stats = df.groupby('year')['temperature'].agg(['min', 'max', 'mean']).reset_index()
    plt.fill_between(yearly_stats['year'], yearly_stats['min'], yearly_stats['max'], 
                     alpha=0.3, label='Min-Max Range')
    plt.plot(yearly_stats['year'], yearly_stats['mean'], color='red', marker='o', label='Average')
    plt.title('Yearly Temperature Range')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    
    # 10. City comparison - recent year
    plt.subplot(3, 4, 10)
    recent_data = df[df['year'] == 2023]
    monthly_recent = recent_data.groupby(['month', 'city'])['temperature'].mean().reset_index()
    for city in df['city'].unique():
        city_recent = monthly_recent[monthly_recent['city'] == city]
        plt.plot(city_recent['month'], city_recent['temperature'], marker='o', label=city)
    plt.title('2023 Monthly Temperatures by City')
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    
    # 11. Temperature volatility
    plt.subplot(3, 4, 11)
    yearly_std = df.groupby('year')['temperature'].std()
    plt.plot(yearly_std.index, yearly_std.values, marker='o', color='purple')
    plt.title('Temperature Volatility Over Years')
    plt.xlabel('Year')
    plt.ylabel('Temperature Std Dev (°C)')
    
    # 12. Correlation between cities
    plt.subplot(3, 4, 12)
    city_pivot = df.pivot_table(index='date', columns='city', values='temperature')
    city_corr = city_pivot.corr()
    sns.heatmap(city_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Temperature Correlation Between Cities')
    
    plt.tight_layout()
    plt.savefig('temperature_timeseries_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Additional detailed analysis
def detailed_timeseries_analysis(df):
    plt.figure(figsize=(16, 10))
    
    # 1. Decomposition-style plot for one city
    plt.subplot(2, 3, 1)
    ny_data = df[df['city'] == 'New York'].copy().sort_values('date')
    
    # Trend component (yearly average)
    ny_yearly = ny_data.groupby('year')['temperature'].mean()
    trend_values = []
    for date in ny_data['date']:
        trend_values.append(ny_yearly[date.year])
    
    plt.plot(ny_data['date'], trend_values, label='Trend', linewidth=2)
    plt.plot(ny_data['date'], ny_data['temperature'], alpha=0.3, label='Actual')
    plt.title('Temperature Trend Analysis - New York')
    plt.legend()
    plt.xticks(rotation=45)
    
    # 2. Climate change visualization
    plt.subplot(2, 3, 2)
    decade_avg = df.groupby([df['year'] // 10 * 10, 'city'])['temperature'].mean().reset_index()
    decade_avg['decade'] = decade_avg['year'].astype(str) + 's'
    
    for city in ['New York', 'London']:
        city_decade = decade_avg[decade_avg['city'] == city]
        plt.bar(city_decade['decade'], city_decade['temperature'], 
                alpha=0.7, label=city, width=0.4)
    plt.title('Decadal Average Temperatures')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    
    # 3. Extreme temperature events
    plt.subplot(2, 3, 3)
    # Define extreme temperatures (beyond 2 standard deviations)
    temp_mean = df['temperature'].mean()
    temp_std = df['temperature'].std()
    
    extreme_hot = df[df['temperature'] > temp_mean + 2*temp_std]
    extreme_cold = df[df['temperature'] < temp_mean - 2*temp_std]
    
    extreme_yearly = df.groupby('year').apply(
        lambda x: len(x[(x['temperature'] > temp_mean + 2*temp_std) | 
                       (x['temperature'] < temp_mean - 2*temp_std)])
    )
    
    plt.bar(extreme_yearly.index, extreme_yearly.values, color='orange')
    plt.title('Extreme Temperature Events per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Extreme Days')
    
    # 4. Seasonal shift analysis
    plt.subplot(2, 3, 4)
    # Calculate when each year reaches peak summer temperature
    summer_peaks = []
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        summer_data = year_data[year_data['season'] == 'Summer']
        if not summer_data.empty:
            peak_day = summer_data.loc[summer_data['temperature'].idxmax(), 'day_of_year']
            summer_peaks.append({'year': year, 'peak_day': peak_day})
    
    peak_df = pd.DataFrame(summer_peaks)
    plt.scatter(peak_df['year'], peak_df['peak_day'], alpha=0.7)
    plt.title('Summer Temperature Peak Timing')
    plt.xlabel('Year')
    plt.ylabel('Day of Year')
    
    # 5. Temperature gradient between cities
    plt.subplot(2, 3, 5)
    city_pairs = [('Mumbai', 'London'), ('Tokyo', 'Sydney'), ('New York', 'Mumbai')]
    
    for i, (city1, city2) in enumerate(city_pairs):
        city1_data = df[df['city'] == city1].groupby('year')['temperature'].mean()
        city2_data = df[df['city'] == city2].groupby('year')['temperature'].mean()
        gradient = city1_data - city2_data
        plt.plot(gradient.index, gradient.values, marker='o', label=f'{city1} - {city2}')
    
    plt.title('Temperature Gradients Between Cities')
    plt.xlabel('Year')
    plt.ylabel('Temperature Difference (°C)')
    plt.legend()
    
    # 6. Forecast visualization (simple linear trend)
    plt.subplot(2, 3, 6)
    global_yearly = df.groupby('year')['temperature'].mean()
    
    # Fit linear trend
    years = global_yearly.index.values
    temps = global_yearly.values
    z = np.polyfit(years, temps, 1)
    p = np.poly1d(z)
    
    # Extend to future years
    future_years = np.arange(2014, 2030)
    future_temps = p(future_years)
    
    plt.plot(years, temps, 'o-', label='Historical', color='blue')
    plt.plot(future_years[len(years):], future_temps[len(years):], 
             '--', label='Linear Projection', color='red')
    plt.title('Temperature Trend Projection')
    plt.xlabel('Year')
    plt.ylabel('Global Average Temperature (°C)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('detailed_temperature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate insights
def generate_timeseries_insights(df):
    insights = []
    
    # Overall trend
    yearly_avg = df.groupby('year')['temperature'].mean()
    temp_trend = yearly_avg.iloc[-1] - yearly_avg.iloc[0]
    insights.append(f"Overall temperature trend: {temp_trend:+.2f}°C over 10 years")
    
    # Warmest and coldest cities
    city_avg = df.groupby('city')['temperature'].mean().sort_values(ascending=False)
    insights.append(f"Warmest city: {city_avg.index[0]} ({city_avg.iloc[0]:.1f}°C avg)")
    insights.append(f"Coldest city: {city_avg.index[-1]} ({city_avg.iloc[-1]:.1f}°C avg)")
    
    # Seasonal variation
    seasonal_range = df.groupby('season')['temperature'].mean()
    max_season = seasonal_range.idxmax()
    min_season = seasonal_range.idxmin()
    insights.append(f"Warmest season: {max_season} ({seasonal_range[max_season]:.1f}°C)")
    insights.append(f"Coldest season: {min_season} ({seasonal_range[min_season]:.1f}°C)")
    
    # Temperature volatility
    yearly_std = df.groupby('year')['temperature'].std().mean()
    insights.append(f"Average yearly temperature volatility: {yearly_std:.2f}°C")
    
    # Extreme temperatures
    temp_mean = df['temperature'].mean()
    temp_std = df['temperature'].std()
    extreme_days = len(df[abs(df['temperature'] - temp_mean) > 2*temp_std])
    insights.append(f"Extreme temperature days (>2σ): {extreme_days} ({extreme_days/len(df)*100:.1f}%)")
    
    return insights

def main():
    print("🌡️ Temperature Time Series Analysis")
    print("="*50)
    
    # Generate temperature data
    df = create_temperature_data()
    print(f"📊 Dataset created: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Cities: {', '.join(df['city'].unique())}")
    
    # Create visualizations
    print("\n📈 Creating time series visualizations...")
    create_timeseries_plots(df)
    
    print("\n🔍 Creating detailed analysis...")
    detailed_timeseries_analysis(df)
    
    # Generate insights
    insights = generate_timeseries_insights(df)
    print("\n" + "="*50)
    print("TIME SERIES INSIGHTS:")
    print("="*50)
    for insight in insights:
        print(f"• {insight}")
    
    # Save data
    df.to_csv('temperature_timeseries_data.csv', index=False)
    
    # Save summary statistics
    summary_stats = df.groupby(['city', 'year'])['temperature'].agg(['mean', 'min', 'max', 'std']).reset_index()
    summary_stats.to_csv('temperature_summary_stats.csv', index=False)
    
    print(f"\n✅ Analysis complete! Check PNG files for visualizations")

if __name__ == "__main__":
    main()