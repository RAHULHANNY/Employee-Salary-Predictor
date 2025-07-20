import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# --- Load Data ---
try:
    df = pd.read_csv('salary_data_regression.csv')
except FileNotFoundError:
    print("❌ 'salary_data_regression.csv' not found. Please run 'generate_data.py' first.")
    exit()

# --- Feature Engineering ---
X = df.drop('Salary', axis=1)
y = df['Salary']

# Identify categorical and numerical features
categorical_features = ['Gender', 'Education Level', 'Job Title']
numerical_features = ['Age', 'Years of Experience']

# --- Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- Model Pipeline ---
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', LinearRegression())])

# --- Train the Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# --- Evaluate the Model ---
y_pred = model_pipeline.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

# Store metrics for the app
performance_metrics = {
    "Mean Absolute Error (MAE)": f"{mae:,.2f}",
    "Mean Squared Error (MSE)": f"{mse:,.2f}",
    "R-squared (R²)": f"{r2:.4f}"
}

# --- Generate and Save Plots ---

# 1. Feature Importance Plot
try:
    # Get feature names after one-hot encoding
    feature_names = numerical_features + \
                    list(model_pipeline.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .get_feature_names_out(categorical_features))
    
    coefficients = model_pipeline.named_steps['regressor'].coef_
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))
    importance = pd.Series(coefficients, index=feature_names).sort_values(key=abs)
    importance.plot(kind='barh', ax=ax, color='#00d9ff')
    ax.set_title('Feature Importance (Coefficient Magnitudes)', fontsize=16)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', transparent=True)
    print("✅ Feature importance plot saved.")

except Exception as e:
    print(f"Could not generate feature importance plot: {e}")


# 2. Scatter Plot: Actual vs. Predicted
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.5, color='#00d9ff', edgecolors='w', linewidth=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Salary', fontsize=12)
ax.set_ylabel('Predicted Salary', fontsize=12)
ax.set_title('Actual vs. Predicted Salary', fontsize=16)
ax.grid(linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_plot.png', transparent=True)
print("✅ Scatter plot saved.")


# 3. Line Plot (Residuals)
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
residuals = y_test - y_pred
ax.scatter(y_pred, residuals, alpha=0.5, color='#00d9ff', edgecolors='w', linewidth=0.5)
ax.hlines(0, y_pred.min(), y_pred.max(), colors='r', linestyles='--')
ax.set_xlabel('Predicted Salary', fontsize=12)
ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
ax.set_title('Residual Plot', fontsize=16)
ax.grid(linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('line_plot.png', transparent=True)
print("✅ Line plot (residuals) saved.")


# --- Save the Model and Metrics ---
joblib.dump(model_pipeline, 'regression_model.pkl')
joblib.dump(performance_metrics, 'performance_metrics.pkl')

print("\n✅ Model, metrics, and plots saved successfully!")
