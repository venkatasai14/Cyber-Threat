from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

app = Flask(__name__)

# Load your CSV data
data = pd.read_csv('minidata.csv', encoding='latin1')

# Extracting URL data into a list or Series
url_data = data['url'].tolist()  # Assuming 'URL' is the column containing URLs
labels = data['type']  # Assuming 'type' is the column containing labels

# Initialize CountVectorizer and transform URLs
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(url_data)

# Create an SVM Classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the model
svm_classifier.fit(X, labels)

# Save the trained model
joblib.dump(svm_classifier, 'svm_modelfinal.joblib')
joblib.dump(vectorizer, 'count_vectorizerfinal.joblib')  # Save the CountVectorizer too

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Load the trained model and vectorizer
        loaded_svm = joblib.load('svm_modelfinal.joblib')
        loaded_vectorizer = joblib.load('count_vectorizerfinal.joblib')

        # Get user input
        user_input_url = request.form['url']

        # Transform the new URL using the loaded vectorizer
        url_transformed = loaded_vectorizer.transform([user_input_url])

        # Make prediction using the loaded SVM model
        predicted_label = loaded_svm.predict(url_transformed)[0]

        return render_template('result.html', url=user_input_url, predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
 