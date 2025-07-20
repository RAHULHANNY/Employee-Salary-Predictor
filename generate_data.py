import pandas as pd
import numpy as np

# --- Configuration ---
NUM_SAMPLES = 1000
JOB_TITLES = ["Software Engineer", "Data Scientist", "Project Manager", "Business Analyst", "DevOps Engineer", "UI/UX Designer"]
EDUCATION_LEVELS = ["Bachelor's", "Master's", "PhD"]

# --- Seed for reproducibility ---
np.random.seed(42)

# --- Generate Data ---
data = {
    'Age': np.random.randint(22, 60, size=NUM_SAMPLES),
    'Gender': np.random.choice(['Male', 'Female'], size=NUM_SAMPLES, p=[0.6, 0.4]),
    'Education Level': np.random.choice(EDUCATION_LEVELS, size=NUM_SAMPLES, p=[0.5, 0.4, 0.1]),
    'Job Title': np.random.choice(JOB_TITLES, size=NUM_SAMPLES),
    'Years of Experience': np.random.randint(0, 35, size=NUM_SAMPLES)
}

df = pd.DataFrame(data)

# --- Create a realistic salary based on other features ---
base_salary = 40000
experience_factor = 2500
age_factor = 300
random_noise = np.random.randint(-5000, 5000, size=NUM_SAMPLES)

# Add multipliers for education and job title
education_multiplier = df['Education Level'].map({"Bachelor's": 1.0, "Master's": 1.2, "PhD": 1.5})
job_multiplier = df['Job Title'].map({
    "Software Engineer": 1.1, 
    "Data Scientist": 1.3, 
    "Project Manager": 1.2, 
    "Business Analyst": 1.0, 
    "DevOps Engineer": 1.25, 
    "UI/UX Designer": 0.9
})

df['Salary'] = (base_salary + 
                df['Years of Experience'] * experience_factor + 
                df['Age'] * age_factor + 
                random_noise) * education_multiplier * job_multiplier

# Ensure salary is a clean integer
df['Salary'] = df['Salary'].astype(int)

# --- Save to CSV ---
df.to_csv('salary_data_regression.csv', index=False)

print("✅ Successfully generated 'salary_data_regression.csv' with 1000 samples.")

