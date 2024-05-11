from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
model = joblib.load("Models/Logistic_Regression_model.joblib")
vectorizer = joblib.load("Models/tfidf_vectorizer.joblib")

# Create Flask app
app = Flask(__name__)

# Render the home page with input form
@app.route("/")
def home():
    return render_template("index.html")

# Define a route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get tweet text from form data
    tweet_text = request.form.get("tweet_text")

    # Handle empty input
    if not tweet_text:
        return render_template("index.html", error="Please provide tweet text.")

    # Vectorize the tweet text using the loaded TF-IDF vectorizer
    tweet_vectorized = vectorizer.transform([tweet_text])

    try:
        # Print the shape and content of tweet_vectorized for debugging
        print("Tweet Vectorized Shape:", tweet_vectorized.shape)
        print("Tweet Vectorized Content:", tweet_vectorized)

        # Make predictions using the loaded model
        prediction = model.predict(tweet_vectorized)[0]

        # Provide clear instructions and feedback on the classification result
        if prediction == 1:
            result = "This tweet indicates a Disaster."
        else:
            result = "This tweet indicate a Non-Disaster."

        # Return prediction result
        return render_template("result.html", result=result, tweet_text=tweet_text)

    except Exception as e:
        # Print the exception for debugging
        print("Prediction Error:", e)

        # Handle prediction error
        return render_template("index.html", error="An error occurred during prediction.")

if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app in debug mode