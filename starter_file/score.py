import joblib
import os
import json
import pandas


# The init() method is called once, when the web service starts up.
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.
def init():
    global model
    model_filename = 'best_model_auto_ml.model'
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)
    model = joblib.load(model_path)


# The run() method is called each time a request is made to the scoring API.
# This will generate a Swagger API document for the web service.
def run(data):
    try:
        data = json.loads(data)['data']
        data = pandas.DataFrame.from_dict(data)
        result = model.predict(data)
        # You can return any JSON-serializable object.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
