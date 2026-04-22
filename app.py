from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        income = float(request.form['income'])
        co_income = float(request.form['co_income'])
        loan_amount = float(request.form['loan_amount'])
        credit_score = float(request.form['credit_score'])

        data = np.array([[income, co_income, loan_amount, credit_score]])

        prediction = model.predict(data)[0]

        if prediction == 1:
            result = "Approved"
        else:
            result = "Rejected"

        return render_template("index.html", result=result)

    except:
        return render_template("index.html", result="Error")

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))