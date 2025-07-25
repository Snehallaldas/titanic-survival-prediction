import pickle
import numpy as np
import pandas as pd
from flask import Flask,render_template, jsonify, request
from src.logger import get_logger

logger = get_logger(__name__)

from alibi_detect.cd import KSDrift
from src.feature_store import RedisFeatureStore
from sklearn.preprocessing import StandardScaler

from prometheus_client import start_http_server,Counter,Gauge

app = Flask(__name__, template_folder="templates") 

prediction_count = Counter('prediction_count', "Number of prediction count")
drift_count = Counter('drift_count', "Number of times drift is detected")

MODEL_PATH = "artifacts/models/random_forest_model.pkl"
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

FEATURE_NAMES = ['Age', 'Fare', 'Sex', 'Pclass', 'Embarked', 'Familysize', 'Isalone',
       'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare']

feature_store = RedisFeatureStore()
scaler = StandardScaler()

def fit_scaler_on_ref_data():
    entity_ids = feature_store.get_all_entity_ids()
    all_features = feature_store.get_batch_features(entity_ids)

    all_features_df = pd.DataFrame.from_dict(all_features, orient='index')[FEATURE_NAMES]

    scaler.fit(all_features_df)

    return scaler.transform(all_features_df)

historical_data = fit_scaler_on_ref_data()
ksd = KSDrift(x_ref=historical_data, p_val=0.05)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        Age = float(data["Age"])
        Fare = float(data["Fare"])
        Sex = int(data["Sex"])
        Pclass = int(data["Pclass"])
        Embarked = int(data["Embarked"])
        Familysize = int(data["Familysize"])
        Isalone = int(data["Isalone"])
        HasCabin = int(data["HasCabin"])
        Title = int(data["Title"])
        Pclass_Fare = int(data["Pclass_Fare"])
        Age_Fare = float(data["Age_Fare"])


        features = pd.DataFrame([[Age, Fare, Sex, Pclass, Embarked, Familysize,Isalone, HasCabin, Title, Pclass_Fare, Age_Fare]],columns=FEATURE_NAMES)

        ### Data Drift Detection ###
        features_scaled = scaler.transform(features)

        drift = ksd.predict(features_scaled)
        print("Drift Response :",drift)
        
        drift_response = drift.get('data',[])
        is_drift = drift_response.get('is_drift', None)

        if is_drift is not None and is_drift==1:
            print("Drift Detected")
            logger.info("Drift Detected")

            drift_count.inc()



        predictions = model.predict(features)[0]
        prediction_count.inc()

        result = 'Survived' if predictions == 1 else 'Did Not Survived'

        return render_template('index.html', prediction_text = f"The prediction is : {result}")
    
    except Exception as e:
        return jsonify({'error' : str(e)})
    
@app.route('/metrics')
def metrics():
    from prometheus_client import generate_latest
    from flask import Response

    return Response(generate_latest(), content_type='text/plain')
    
if __name__ =="__main__":
    start_http_server(8000)
    app.run(debug=True, host='0.0.0.0', port=5000)