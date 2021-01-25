# app/app.py

# Common python package imports.
from flask import Flask, jsonify, request, render_template, make_response
from flask_cors import CORS, cross_origin
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Import from model_api/app/features.py.
from features import FEATURES


# Initialize the app and set a secret_key.
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.secret_key = 'something_secret'

clf = joblib.load('covid19_model.pkl')


@app.route('/')
@cross_origin()
def docs():
    """Describe the model API inputs and outputs for users."""
    return render_template('docs.html')


@app.route('/api', methods=['POST'])
@cross_origin()
def api():
    """Handle request and output model score in json format."""
    # Handle empty requests.
    if not request.json:
        return jsonify({'error': 'no request received'})

    # Parse request args into feature array for prediction.
    x_list, missing_data = parse_args(request.json)
    x_array = np.array([x_list])

    predicted = clf.predict_proba(x_array)
    # rf = 1 if (predicted[0, 1] >= 0.5) else 0
    # etc = 1 if (predicted[0, 3] >= 0.5) else 0
    # mlp = 1 if (predicted[0, 5] >= 0.5) else 0
    # svm = 1 if (predicted[0, 7] >= 0.5) else 0
    # percentages = [predicted[0, 1],predicted[0, 3],predicted[0, 5],predicted[0, 7]]
    percentages = [predicted[0, 0],predicted[0, 1]]
    # predicted = predicted[0, 1] + predicted[0, 3] + predicted[0, 5] + predicted[0, 7]

    # predicted = 1 if ((rf+etc+mlp+svm+gnb) > 2) else 0
    # predicted = 1 if (predicted[0, 1] >= 0.59) else 0
    # predicted = 1 if (predicted >= 1.6) else 0
    predicted = predicted[0, 1]

    # if (x_array[0][18] == 1 and x_array[0][19] == 1 and x_array[0][8] == 1):
    #     predicted = 1

    response = dict(PREDICTED=predicted, MISSING_FEATURES=missing_data, PERCENTAGES=percentages)
    return jsonify(response)

@app.route('/loaderio-233ff5ec6246b50a75ee000d61547336/', methods=['GET'])
@cross_origin()
def token():
    """Token for loader.io."""
    response = make_response('loaderio-233ff5ec6246b50a75ee000d61547336', 200)
    response.mimetype = "text/plain"
    return response

def parse_args(request_dict):
    """Parse model features from incoming requests formatted in JSON."""
    # Initialize missing_data as False.
    missing_data = False

    # Parse out the features from the request_dict.
    x_list = []
    for feature in FEATURES:
        value = request_dict.get(feature, None)
        if value:
            x_list.append(float(value))
        else:
            # Handle missing features.
            x_list.append(0)
            missing_data = True
    return x_list, missing_data


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8721, debug=True)
