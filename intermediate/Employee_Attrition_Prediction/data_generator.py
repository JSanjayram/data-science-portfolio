import pandas as pd
import numpy as np

def generate_sample_data(n_samples=1000):
    """Generate sample HR data for demonstration"""
    np.random.seed(42)
    
    # Generate base features
    ages = np.random.normal(35, 8, n_samples).astype(int)
    ages = np.clip(ages, 18, 65)
    
    monthly_income = np.random.normal(5000, 2000, n_samples)
    monthly_income = np.clip(monthly_income, 2000, 15000)
    
    years_at_company = np.random.exponential(5, n_samples)
    years_at_company = np.clip(years_at_company, 0, 30)
    
    # Categorical features
    job_roles = np.random.choice(['Sales Executive', 'Research Scientist', 'Laboratory Technician', 
                                 'Manufacturing Director', 'Healthcare Representative', 'Manager'], n_samples)
    departments = np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n_samples)
    overtime = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    
    # Satisfaction scores (1-4 scale)
    job_satisfaction = np.random.randint(1, 5, n_samples)
    work_life_balance = np.random.randint(1, 5, n_samples)
    
    # Generate attrition based on logical rules
    attrition_prob = (
        (ages < 25) * 0.3 +  # Young employees more likely to leave
        (monthly_income < 3000) * 0.4 +  # Low income increases attrition
        (overtime == 'Yes') * 0.3 +  # Overtime increases attrition
        (job_satisfaction <= 2) * 0.5 +  # Low satisfaction increases attrition
        (years_at_company < 2) * 0.3  # New employees more likely to leave
    )
    
    attrition_prob = np.clip(attrition_prob, 0.05, 0.8)
    attrition = np.random.binomial(1, attrition_prob, n_samples)
    attrition = ['Yes' if x == 1 else 'No' for x in attrition]
    
    # Create DataFrame
    data = pd.DataFrame({
        'Age': ages,
        'MonthlyIncome': monthly_income.astype(int),
        'YearsAtCompany': years_at_company.round(1),
        'JobRole': job_roles,
        'Department': departments,
        'OverTime': overtime,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'Attrition': attrition
    })
    
    return data

if __name__ == "__main__":
    # Generate and save sample data
    df = generate_sample_data(1000)
    df.to_csv('employee_data.csv', index=False)
    print("Sample data generated and saved to employee_data.csv")