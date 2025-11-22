import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate student performance data
def create_student_data():
    np.random.seed(42)
    n_students = 500
    
    # Base attributes
    study_hours = np.random.normal(5, 2, n_students).clip(0, 12)
    attendance = np.random.normal(85, 10, n_students).clip(50, 100)
    sleep_hours = np.random.normal(7, 1.5, n_students).clip(4, 10)
    
    # Correlated performance metrics
    math_score = (study_hours * 8 + attendance * 0.5 + sleep_hours * 3 + np.random.normal(0, 10, n_students)).clip(0, 100)
    science_score = (study_hours * 7 + attendance * 0.6 + sleep_hours * 2.5 + np.random.normal(0, 12, n_students)).clip(0, 100)
    english_score = (study_hours * 6 + attendance * 0.7 + sleep_hours * 2 + np.random.normal(0, 15, n_students)).clip(0, 100)
    
    # Additional factors
    family_income = np.random.lognormal(10, 0.5, n_students)
    extracurricular = np.random.randint(0, 6, n_students)
    screen_time = np.random.normal(4, 2, n_students).clip(0, 12)
    
    # Overall GPA calculation
    gpa = (math_score + science_score + english_score) / 30  # Scale to 4.0
    
    data = {
        'student_id': range(1, n_students + 1),
        'study_hours_per_day': study_hours,
        'attendance_percentage': attendance,
        'sleep_hours': sleep_hours,
        'math_score': math_score,
        'science_score': science_score,
        'english_score': english_score,
        'gpa': gpa,
        'family_income': family_income,
        'extracurricular_activities': extracurricular,
        'screen_time_hours': screen_time,
        'total_score': math_score + science_score + english_score
    }
    
    return pd.DataFrame(data)

# Create correlation heatmaps
def create_correlation_analysis(df):
    plt.figure(figsize=(16, 12))
    
    # 1. Full correlation heatmap
    plt.subplot(2, 3, 1)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Complete Correlation Matrix')
    
    # 2. Academic performance correlation
    plt.subplot(2, 3, 2)
    academic_cols = ['math_score', 'science_score', 'english_score', 'gpa', 'total_score']
    academic_corr = df[academic_cols].corr()
    sns.heatmap(academic_corr, annot=True, cmap='Greens', 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Academic Performance Correlation')
    
    # 3. Lifestyle factors correlation
    plt.subplot(2, 3, 3)
    lifestyle_cols = ['study_hours_per_day', 'sleep_hours', 'screen_time_hours', 'attendance_percentage']
    lifestyle_corr = df[lifestyle_cols].corr()
    sns.heatmap(lifestyle_corr, annot=True, cmap='Blues', 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Lifestyle Factors Correlation')
    
    # 4. Performance vs factors
    plt.subplot(2, 3, 4)
    factor_cols = ['study_hours_per_day', 'attendance_percentage', 'sleep_hours', 'gpa']
    factor_corr = df[factor_cols].corr()
    sns.heatmap(factor_corr, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Key Performance Factors')
    
    # 5. Clustermap
    plt.subplot(2, 3, 5)
    performance_cols = ['math_score', 'science_score', 'english_score', 'study_hours_per_day', 'attendance_percentage']
    perf_corr = df[performance_cols].corr()
    sns.heatmap(perf_corr, annot=True, cmap='viridis',
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Performance Cluster Analysis')
    
    # 6. Strong correlations only
    plt.subplot(2, 3, 6)
    strong_corr = corr_matrix.copy()
    strong_corr[abs(strong_corr) < 0.3] = 0  # Show only strong correlations
    sns.heatmap(strong_corr, annot=True, cmap='RdYlBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Strong Correlations (|r| > 0.3)')
    
    plt.tight_layout()
    plt.savefig('student_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Additional analysis
def detailed_correlation_analysis(df):
    plt.figure(figsize=(15, 10))
    
    # 1. Scatter plot matrix for key variables
    plt.subplot(2, 3, 1)
    key_vars = ['gpa', 'study_hours_per_day', 'attendance_percentage', 'sleep_hours']
    sns.scatterplot(data=df, x='study_hours_per_day', y='gpa', alpha=0.6)
    plt.title('GPA vs Study Hours')
    
    plt.subplot(2, 3, 2)
    sns.scatterplot(data=df, x='attendance_percentage', y='gpa', alpha=0.6, color='orange')
    plt.title('GPA vs Attendance')
    
    plt.subplot(2, 3, 3)
    sns.scatterplot(data=df, x='sleep_hours', y='gpa', alpha=0.6, color='green')
    plt.title('GPA vs Sleep Hours')
    
    # 4. Subject correlation comparison
    plt.subplot(2, 3, 4)
    subject_corr = df[['math_score', 'science_score', 'english_score']].corr()
    sns.heatmap(subject_corr, annot=True, cmap='Oranges', square=True, fmt='.3f')
    plt.title('Inter-Subject Correlations')
    
    # 5. Performance distribution
    plt.subplot(2, 3, 5)
    df['performance_category'] = pd.cut(df['gpa'], bins=3, labels=['Low', 'Medium', 'High'])
    sns.countplot(data=df, x='performance_category', palette='viridis')
    plt.title('Performance Distribution')
    
    # 6. Factor importance heatmap
    plt.subplot(2, 3, 6)
    factors = ['study_hours_per_day', 'attendance_percentage', 'sleep_hours', 'extracurricular_activities']
    factor_impact = df[factors + ['gpa']].corr()['gpa'].drop('gpa')
    factor_df = pd.DataFrame({'Factor': factor_impact.index, 'Correlation': factor_impact.values})
    sns.barplot(data=factor_df, x='Correlation', y='Factor', palette='rocket')
    plt.title('Factor Impact on GPA')
    
    plt.tight_layout()
    plt.savefig('detailed_student_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate correlation insights
def generate_correlation_insights(df):
    insights = []
    
    # Calculate key correlations
    corr_matrix = df.corr()
    
    # Strongest positive correlation with GPA
    gpa_corr = corr_matrix['gpa'].drop('gpa').sort_values(ascending=False)
    strongest_pos = gpa_corr.iloc[0]
    insights.append(f"Strongest positive factor: {gpa_corr.index[0]} (r={strongest_pos:.3f})")
    
    # Subject correlations
    math_science = corr_matrix.loc['math_score', 'science_score']
    math_english = corr_matrix.loc['math_score', 'english_score']
    science_english = corr_matrix.loc['science_score', 'english_score']
    
    insights.append(f"Math-Science correlation: {math_science:.3f}")
    insights.append(f"Math-English correlation: {math_english:.3f}")
    insights.append(f"Science-English correlation: {science_english:.3f}")
    
    # Lifestyle impact
    study_gpa = corr_matrix.loc['study_hours_per_day', 'gpa']
    sleep_gpa = corr_matrix.loc['sleep_hours', 'gpa']
    attendance_gpa = corr_matrix.loc['attendance_percentage', 'gpa']
    
    insights.append(f"Study hours impact on GPA: {study_gpa:.3f}")
    insights.append(f"Sleep impact on GPA: {sleep_gpa:.3f}")
    insights.append(f"Attendance impact on GPA: {attendance_gpa:.3f}")
    
    return insights

def main():
    print("📊 Student Performance Correlation Analysis")
    print("="*50)
    
    # Generate data
    df = create_student_data()
    print(f"📈 Dataset created: {df.shape}")
    
    # Create correlation analysis
    print("🔍 Creating correlation heatmaps...")
    create_correlation_analysis(df)
    
    print("📋 Creating detailed analysis...")
    detailed_correlation_analysis(df)
    
    # Generate insights
    insights = generate_correlation_insights(df)
    print("\n" + "="*50)
    print("CORRELATION INSIGHTS:")
    print("="*50)
    for insight in insights:
        print(f"• {insight}")
    
    # Save data
    df.to_csv('student_performance_data.csv', index=False)
    print(f"\n✅ Analysis complete! Check PNG files for visualizations")

if __name__ == "__main__":
    main()