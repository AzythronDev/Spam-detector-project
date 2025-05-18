from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    if request.method == 'POST':
        message = request.form['message']
        message_vector = vectorizer.transform([message])
        pred = model.predict(message_vector)
        prediction = "Spam" if pred[0] == 1 else "Ham (Not Spam)"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
