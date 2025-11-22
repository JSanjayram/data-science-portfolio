import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

# Load Titanic dataset
def load_data():
    titanic = fetch_openml('titanic', version=1, as_frame=True)
    df = titanic.frame
    # Convert categorical survived to numeric
    df['survived'] = df['survived'].astype(int)
    return df

# Basic data exploration
def explore_data(df):
    print("Dataset Shape:", df.shape)
    print("\nColumn Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nBasic Statistics:")
    print(df.describe())

# Survival analysis
def survival_analysis(df):
    plt.figure(figsize=(15, 10))
    
    # Survival rate
    plt.subplot(2, 3, 1)
    df['survived'].value_counts().plot(kind='bar')
    plt.title('Survival Count')
    
    # Survival by class
    plt.subplot(2, 3, 2)
    pd.crosstab(df['pclass'], df['survived']).plot(kind='bar')
    plt.title('Survival by Class')
    
    # Survival by sex
    plt.subplot(2, 3, 3)
    pd.crosstab(df['sex'], df['survived']).plot(kind='bar')
    plt.title('Survival by Gender')
    
    # Age distribution
    plt.subplot(2, 3, 4)
    df['age'].hist(bins=30)
    plt.title('Age Distribution')
    
    # Fare distribution
    plt.subplot(2, 3, 5)
    df['fare'].hist(bins=30)
    plt.title('Fare Distribution')
    
    # Survival by age groups
    plt.subplot(2, 3, 6)
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young', 'Adult', 'Senior'])
    pd.crosstab(df['age_group'], df['survived']).plot(kind='bar')
    plt.title('Survival by Age Group')
    
    plt.tight_layout()
    plt.savefig('titanic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Key insights
def generate_insights(df):
    insights = []
    
    # Overall survival rate
    survival_rate = df['survived'].mean()
    insights.append(f"Overall survival rate: {survival_rate:.2%}")
    
    # Survival by gender
    gender_survival = df.groupby('sex')['survived'].mean()
    insights.append(f"Female survival rate: {gender_survival['female']:.2%}")
    insights.append(f"Male survival rate: {gender_survival['male']:.2%}")
    
    # Survival by class
    class_survival = df.groupby('pclass')['survived'].mean()
    insights.append(f"1st class survival: {class_survival[1]:.2%}")
    insights.append(f"2nd class survival: {class_survival[2]:.2%}")
    insights.append(f"3rd class survival: {class_survival[3]:.2%}")
    
    return insights

def main():
    # Load and explore data
    df = load_data()
    explore_data(df)
    
    # Generate visualizations
    survival_analysis(df)
    
    # Print insights
    insights = generate_insights(df)
    print("\n" + "="*50)
    print("KEY INSIGHTS:")
    print("="*50)
    for insight in insights:
        print(f"• {insight}")

if __name__ == "__main__":
    main()