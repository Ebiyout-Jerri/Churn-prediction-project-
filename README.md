# Churn_Prediction_Project
```markdown
# Customer Churn Prediction Web App

- A machine learning-powered web application built using **Python** and **Streamlit**
- It predicts whether a customer will churn based on their input attributes.
- This project showcases data preprocessing, model training, deployment, and a sleek interactive UI.

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, scikit-learn, ReportLab, 
- **Visualization**: Seaborn, Matplotlib, SHAP
- **Deployment**: Localhost

## How It Works

1. Users input customer details via the sidebar form.
2. The trained ML model processes the data.
3. Churn prediction and confidence score are displayed.
4. Visual feedback explains key influencing features.
```
## Project Structure
```
churn-prediction-app/
│
├── dashboard.py                  # Streamlit UI code
├── best_model.pkl                # Trained machine learning model
├── Churn_Modelling.csv           # dataset
├── pipeline.pkl                  # Data preprocessing pipeline
└── churn_prediction.ipynb        # EDA and model building notebook
```
## Prediction Output
```
The app outputs:
- **Customer input**
- **Prediction**: Churn / No Churn
- **Probability Score**
- **Retention Strategy**
- **Feature Importance Summary Visuals**
```
## Enhancements
```
- Integrated SHAP for explainability
```
Made to help businesses keep their valued customers.
