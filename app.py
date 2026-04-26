import pickle
from flask import Flask, request, app, jsonify, render_template
import numpy as np

app=Flask(__name__)

# Load The Model
scaler_model = pickle.load(open("scaler.pkl", "rb"))
predict_model = pickle.load(open("predict-price-of-house.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    transformed_data = scaler_model.transform(np.array(list(data.values())).reshape(1, -1))
    output = predict_model.predict(transformed_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler_model.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = predict_model.predict(final_input)[0]
    return render_template("home.html", text="The Predicted House Price is {} Lakhs".format(output))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

