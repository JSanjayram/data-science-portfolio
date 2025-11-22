import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load COVID-19 data
def load_covid_data():
    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Global trends analysis
def analyze_global_trends(df):
    global_data = df.groupby('date').agg({
        'new_cases': 'sum',
        'new_deaths': 'sum',
        'total_cases': 'sum',
        'total_deaths': 'sum'
    }).reset_index()
    
    plt.figure(figsize=(15, 10))
    
    # Daily new cases
    plt.subplot(2, 2, 1)
    plt.plot(global_data['date'], global_data['new_cases'])
    plt.title('Global Daily New Cases')
    plt.xticks(rotation=45)
    
    # Daily new deaths
    plt.subplot(2, 2, 2)
    plt.plot(global_data['date'], global_data['new_deaths'], color='red')
    plt.title('Global Daily New Deaths')
    plt.xticks(rotation=45)
    
    # Cumulative cases
    plt.subplot(2, 2, 3)
    plt.plot(global_data['date'], global_data['total_cases'], color='orange')
    plt.title('Global Total Cases')
    plt.xticks(rotation=45)
    
    # Cumulative deaths
    plt.subplot(2, 2, 4)
    plt.plot(global_data['date'], global_data['total_deaths'], color='purple')
    plt.title('Global Total Deaths')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('covid19_global_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return global_data

# Top countries analysis
def analyze_top_countries(df):
    latest_data = df.groupby('location')['total_cases'].max().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(12, 8))
    
    # Top 10 countries by cases
    plt.subplot(2, 1, 1)
    latest_data.plot(kind='bar')
    plt.title('Top 10 Countries by Total Cases')
    plt.xticks(rotation=45)
    
    # Deaths comparison
    plt.subplot(2, 1, 2)
    top_countries = latest_data.index
    death_data = df.groupby('location')['total_deaths'].max().loc[top_countries]
    death_data.plot(kind='bar', color='red')
    plt.title('Total Deaths in Top 10 Countries')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('covid19_countries.png', dpi=300, bbox_inches='tight')
    plt.show()

# Monthly trends
def analyze_monthly_trends(df):
    df['month'] = df['date'].dt.to_period('M')
    monthly_data = df.groupby('month').agg({
        'new_cases': 'sum',
        'new_deaths': 'sum'
    }).reset_index()
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(monthly_data)), monthly_data['new_cases'])
    plt.title('Monthly New Cases')
    plt.xticks(range(0, len(monthly_data), 3), 
               [str(monthly_data['month'].iloc[i]) for i in range(0, len(monthly_data), 3)], 
               rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(monthly_data)), monthly_data['new_deaths'], color='red')
    plt.title('Monthly New Deaths')
    plt.xticks(range(0, len(monthly_data), 3), 
               [str(monthly_data['month'].iloc[i]) for i in range(0, len(monthly_data), 3)], 
               rotation=45)
    
    plt.tight_layout()
    plt.savefig('covid19_monthly.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate insights
def generate_insights(df, global_data):
    insights = []
    
    # Peak cases day
    peak_day = global_data.loc[global_data['new_cases'].idxmax()]
    insights.append(f"Peak daily cases: {peak_day['new_cases']:,.0f} on {peak_day['date'].strftime('%Y-%m-%d')}")
    
    # Total statistics
    total_cases = global_data['total_cases'].iloc[-1]
    total_deaths = global_data['total_deaths'].iloc[-1]
    insights.append(f"Total global cases: {total_cases:,.0f}")
    insights.append(f"Total global deaths: {total_deaths:,.0f}")
    insights.append(f"Global mortality rate: {(total_deaths/total_cases)*100:.2f}%")
    
    # Most affected country
    most_affected = df.groupby('location')['total_cases'].max().idxmax()
    max_cases = df.groupby('location')['total_cases'].max().max()
    insights.append(f"Most affected country: {most_affected} ({max_cases:,.0f} cases)")
    
    return insights

def main():
    print("📊 Loading COVID-19 data...")
    df = load_covid_data()
    
    print("🌍 Analyzing global trends...")
    global_data = analyze_global_trends(df)
    
    print("🏆 Analyzing top countries...")
    analyze_top_countries(df)
    
    print("📅 Analyzing monthly trends...")
    analyze_monthly_trends(df)
    
    # Generate insights
    insights = generate_insights(df, global_data)
    print("\n" + "="*50)
    print("KEY INSIGHTS:")
    print("="*50)
    for insight in insights:
        print(f"• {insight}")
    
    print("\n✅ Analysis complete! Check generated PNG files.")

if __name__ == "__main__":
    main()