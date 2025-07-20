SalarySense: AI-Powered Employee Salary Predictor 🧠💸
An interactive web application built with Streamlit that predicts employee salaries using a Linear Regression model. SalarySense provides a user-friendly interface to get salary estimates and offers deep-dive insights into the model's performance and decision-making process.

Key Features & Screenshots
SalarySense provides a clean, intuitive interface for users to input employee details and receive an instant salary prediction.

1. Interactive Prediction Tool
Users can adjust sliders and select options for Age, Job Title, Years of Experience, Gender, and Education Level to generate a salary estimate.
Markdown
![Prediction UI]
![WhatsApp Image 2025-07-20 at 19 20 38_c8f69e12](https://github.com/user-attachments/assets/c3514d6f-d126-4496-b0b7-cb2435a55a47)

3. Instant Salary Output
The model instantly computes and displays the predicted salary based on the inputs.
Markdown
![Prediction Output](![WhatsApp Image 2025-07-20 at 19 20 56_bbdd2f50](https://github.com/user-attachments/assets/5e4a84a6-09ff-4e66-89c5-fdd4e7bd6bf1)
)
4. In-Depth Model Insights
The application includes several visualizations to understand the model's behavior:
Feature Importance: Shows the impact (coefficient) of each feature on the salary prediction.
Actual vs. Predicted Plot: A scatter plot to visualize the model's accuracy.
Residual Plot: Helps diagnose the variance of errors in the model.

Markdown

![Feature Importance](![WhatsApp Image 2025-07-20 at 19 21 18_4335f7bc](https://github.com/user-attachments/assets/221ffdca-f1bd-42b0-8d1f-22dad47dd64c)
)
![Actual vs Predicted](  
Model & Performance
The prediction is powered by a Linear Regression model trained on key employee attributes. The model demonstrates high accuracy and reliability, as shown by the performance metrics below:
![WhatsApp Image 2025-07-20 at 19 21 32_7412a9b5](https://github.com/user-attachments/assets/d5d8dbab-1613-4ce5-b7c3-bed5e5f672dd)

Metric	Value
R-squared (R²)	0.9685
Mean Absolute Error (MAE)	5,913.58
Mean Squared Error (MSE)	56,695,101.59
An R² value of 0.9685 indicates that the model explains approximately 97% of the variance in the salary data, which is an excellent fit.

Tech Stack

Language: Python
Web Framework: Streamlit
ML & Data Libraries: Scikit-learn, Pandas, NumPy
Visualization: Matplotlib, Seaborn
Installation
To get a local copy up and running, follow these steps.

Clone the repository:

Bash
git clone https://github.com/[RAHULHANNY]/[Employee-Salary-Predictor].git
Navigate to the project directory:

Bash
cd [Employee-Salary-Predictor]
Install the required dependencies:

Bash
pip install -r requirements.txt
Usage
To run the web application, execute the following command in your terminal:

Bash
streamlit run app.py
This will start the application, and you can access it in your web browser at the local URL provided 
http://localhost:8501/
Contributing
Contributions are welcome! If you have suggestions to improve the project, please fork the repository and create a pull request, or open an issue with the "enhancement" tag.
