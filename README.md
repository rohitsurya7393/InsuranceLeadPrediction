

# 🚀 Health Insurance Lead Prediction

## 📌 Project Description
The **Health Insurance Lead Prediction** project aims to help insurance companies efficiently identify potential customers through **Machine Learning (ML)** models. The project utilizes **predictive analytics** to analyze user behavior and determine lead conversions. By applying various ML algorithms, we provide insights into customer preferences, enabling insurers to optimize their marketing strategies and improve business growth.

This project is structured into three phases:
1. **Data Preprocessing & Cleaning:** Handled missing values, outliers, and inconsistencies to ensure a high-quality dataset.
2. **Model Training & Evaluation:** Applied multiple ML algorithms to predict lead conversions and evaluated their performance.
3. **Web Application Deployment:** Created an interactive **Flask web application** where users can input real-world data, select an ML model, and obtain predictions.

## 🎯 Key Features
- **Data Cleaning & Feature Engineering**: Imputation, scaling, encoding, and outlier detection.
- **Multiple ML Models**: Logistic Regression, Decision Trees, Random Forest, Neural Networks, etc.
- **User Input via Web App**: A frontend interface allows users to input customer data and receive predictions.
- **Saved ML Models**: Models are serialized using Python’s `pickle` library for future use.
- **Interactive Visualizations**: Model performance analysis with confusion matrices and evaluation metrics.

---

## 🛠️ How to Run the Code

### 🔹 Prerequisites
Ensure you have the following installed:
- Python 3.x
- Flask
- NumPy, Pandas, Scikit-learn
- Pickle (for model saving/loading)

Install dependencies using bash:

pip install flask numpy pandas scikit-learn

## 🔹 Steps to Run
Run Model Training
First, train the ML models and save them as .pkl files:
python train_models.py
This will generate saved models in the models/ directory.
Start the Web Application Navigate to the app directory and run:
python app.py
This will start a Flask server on http://127.0.0.1:5000/
The terminal will display a localhost link.
Access the Web App Open the provided localhost link in a browser. You can:
Enter customer details.
Select the ML model for prediction.
Get instant results on whether the customer is a lead or not.

## 🌍 How to Run the Website

Ensure the Flask app is running (app.py).
Open a web browser and go to:
http://127.0.0.1:5000/
Enter Data & Select Model:
Fill in the customer details.
Choose an ML model for prediction.
Submit & Get Results:
The website processes the input.
The selected model predicts if the input qualifies as an insurance lead.
Results are displayed on the result page.

## 📊 Model Evaluation

The following models were tested for accuracy and performance:

Model	Accuracy
Logistic Regression	76%
Decision Tree	73.58%
Random Forest	74.74%
Neural Network	74.64%
K-Nearest Neighbors (KNN)	74.7%
Naïve Bayes	60.72%
Linear Regression (for classification)	76.04%
K-Means (unsupervised)	33.58%
Evaluation metrics include:

Accuracy, Precision, Recall, F1-Score
Confusion Matrices
Feature Importance Visualizations

## 🔥 Future Enhancements

Real-time Data Ingestion: Automate lead prediction using live data.
CRM Integration: Connect with insurance CRM tools.
Advanced Deep Learning Models: Improve prediction accuracy.

## 📜 References

Flask Documentation: https://flask.palletsprojects.com
Scikit-Learn: https://scikit-learn.org
Pickle Module: https://docs.python.org/3/library/pickle.html
