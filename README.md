# Forest Fire Prediction Model - Azure Machine Learning Studio

Project Overview

This project uses Azure Machine Learning Studio's Automated ML to build a predictive model for forecasting the occurrence and spread of forest fires based on a given dataset. By leveraging Azure's powerful AutoML capabilities, the best-performing model was selected to provide reliable and optimized predictions.

Tools and Technologies

Azure Machine Learning Studio

Automated Machine Learning (AutoML)

Python

Machine Learning Algorithms

Dataset

The dataset used in this project contains historical information related to forest fires, including spatial coordinates, date attributes, and environmental conditions. This data was processed and fed into the AutoML pipeline to determine the relationship between these features and the area affected by forest fires.

Key Features in the Dataset:

X - x-axis spatial coordinate within the Montesinho park map: 1 to 9

Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9

month - month of the year: 'jan' to 'dec'

day - day of the week: 'mon' to 'sun'

FFMC - FFMC index from the FWI system: 18.7 to 96.20

DMC - DMC index from the FWI system: 1.1 to 291.3

DC - DC index from the FWI system: 7.9 to 860.6

ISI - ISI index from the FWI system: 0.0 to 56.10

temp - temperature in Celsius degrees: 2.2 to 33.30

RH - relative humidity in %: 15.0 to 100

wind - wind speed in km/h: 0.40 to 9.40

rain - outside rain in mm/m2: 0.0 to 6.4

area - the burned area of the forest (in ha): 0.00 to 1090.84

(This output variable is very skewed towards 0.0, thus it may make sense to model with the logarithm transform.)

Goal of the Project

The primary goal of this project is to predict the likelihood and scale of forest fires based on environmental conditions. Accurate predictions can help authorities take preventive measures and allocate resources efficiently to mitigate fire damage.

Model Development

Automated ML Configuration:

Task Type: Regression (predicting continuous values such as fire area affected)

Input Dataset: Historical forest fire data

Target Column: Area (size of the affected region)

Model Training:

Azure ML AutoML tested multiple algorithms, optimized hyperparameters, and selected the best model based on performance metrics.

Performance Metrics Evaluated:

Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

Best Model Selection:

The model with the lowest RMSE and MAE was chosen for deployment.

Results

The trained model achieved the following results:

Root Mean Squared Error (RMSE): X.XX

Mean Absolute Error (MAE): X.XX

These metrics indicate the model's accuracy and reliability in predicting the area affected by forest fires.

How to Use This Repository

Clone the Repository:

git clone https://github.com/yourusername/forest-fire-azure-ml.git

Download the Model:

The trained model file (model.pkl) is included in this repository.

Dependencies:
Install the required Python libraries using the following command:

pip install -r requirements.txt

Run the Model:
You can use the saved model for predictions:

import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Example input data (X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain)
input_data = np.array([[6, 5, 'aug', 'fri', 90.0, 35.0, 670.0, 10.0, 25.0, 40.0, 4.0, 0.0]])
prediction = model.predict(input_data)

print("Predicted area affected by fire (ha):", prediction)

Future Improvements

Fine-tune feature engineering for better performance.

Explore additional environmental factors (e.g., vegetation index, air quality).

Deploy the model as an Azure endpoint for real-time predictions.

Author

Matheus Santhiago Bernal Jorge de Oliveira Borges

Let's work together to prevent forest fires through the power of data and machine learning! 🌲🔥

#AzureMachineLearning #AutoML #MachineLearning #ForestFires #DataScience #Python


