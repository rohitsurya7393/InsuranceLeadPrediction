from flask import Flask, request, render_template
import pickle
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Execute the following commands in the terminal if the error says there is an already existing service in port 5000
# sudo lsof -i :5000
# sudo kill -9 <PID>

# List of paths to your machine learning model files
model_paths = [
    ('Decision Tree','/Users/rohithsurya/Documents/Lab/DIC_LAB/SuryaVenkataRohit_BhavaniKiran_phase_2/Phase_3/src/Models/dt.pkl'),
    ('K Means','/Users/rohithsurya/Documents/Lab/DIC_LAB/SuryaVenkataRohit_BhavaniKiran_phase_2/Phase_3/src/Models/kmeans.pkl'),
    ('KNN','/Users/rohithsurya/Documents/Lab/DIC_LAB/SuryaVenkataRohit_BhavaniKiran_phase_2/Phase_3/src/Models/knn.pkl'),
    ('Linear Regression','/Users/rohithsurya/Documents/Lab/DIC_LAB/SuryaVenkataRohit_BhavaniKiran_phase_2/Phase_3/src/Models/LinearReg.pkl'),
    ('Logistic','/Users/rohithsurya/Documents/Lab/DIC_LAB/SuryaVenkataRohit_BhavaniKiran_phase_2/Phase_3/src/Models/LogisticReg.pkl'),
    ('Naive Bayes','/Users/rohithsurya/Documents/Lab/DIC_LAB/SuryaVenkataRohit_BhavaniKiran_phase_2/Phase_3/src/Models/nb_classifier.pkl'),
    ('Neural Network','/Users/rohithsurya/Documents/Lab/DIC_LAB/SuryaVenkataRohit_BhavaniKiran_phase_2/Phase_3/src/Models/nn.pkl'),
    ('Random Forest','/Users/rohithsurya/Documents/Lab/DIC_LAB/SuryaVenkataRohit_BhavaniKiran_phase_2/Phase_3/src/Models/random_forest.pkl')
]



# Loading Machine Learning model from specefic path
models = {}

for name, path in model_paths:
    with open(path, 'rb') as f:
        models[name] = pickle.load(f)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # The input data from the website is retrieved here
    input_data = [float(request.form[f'value{i+1}']) for i in range(20)]
    #print(input_data)
    input_array = np.array([input_data]) 
    temp_mean = np.mean(input_array)
    temp_std = np.std(input_array) if np.std(input_array) != 0 else 1.0  # This condition is to avoid infinite values


# Scaling the data Manually
    scaled_data = (input_array - temp_mean) / temp_std

    #print(scaled_data)

    # Feeding the input data to saved models
    results = {}
    for name, model in models.items():
        # Selected Model from UI
        print(request.form['value21'])
        if   name == request.form['value21']:
            prediction = model.predict(scaled_data)
            accuracy = model.score(scaled_data, prediction)  

            results[name] = (prediction[0], accuracy)

    # Post the Results to the resutl website
    return render_template('result.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
