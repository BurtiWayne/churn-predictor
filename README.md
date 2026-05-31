# churn-predictor
Customer Churn Prediction App
📡 Customer Churn Prediction App
A machine learning web application that predicts whether a telecom customer is likely to churn, built with Python and deployed using Streamlit.
🔗 Live App: Click here to try it

📊 Dataset

Rows: 300 customers
Features: 19 input features
Target: Churn (Yes / No)

Features used:
gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

🤖 Model
ParameterValueAlgorithmRandom Forest Classifiern_estimators200max_depth10min_samples_split4class_weightbalancedrandom_state42Test size20%

🛠️ Tech Stack

Python
Scikit-learn — model training
Pandas — data processing
Streamlit — web app & deployment
GitHub — version control
