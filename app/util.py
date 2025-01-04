import pickle

# Load models only once at the top-level to avoid repeated I/O operations
with open("../models/cv.pkl", "rb") as cv_file, open("../models/clf.pkl", "rb") as clf_file:
    cv = pickle.load(cv_file)
    clf = pickle.load(clf_file)

def model_predict(email):
    """
    Predicts whether an email is spam or not.

    Args:
        email (str): The email content to classify.

    Returns:
        int: 1 if spam, -1 if not spam.
    """
    if not email.strip():  # Check for empty or whitespace-only input
        return -1  # Default to "not spam" for empty input
    
    try:
        # Transform and predict
        tokenized_email = cv.transform([email])  # Convert email to feature vector
        prediction = clf.predict(tokenized_email)[0]  # Get the first prediction
        
        # Return 1 for spam, -1 for not spam
        return 1 if prediction == 1 else -1
    except Exception as e:
        # Log the error if needed and return a neutral prediction
        print(f"Error during prediction: {e}")
        return -1  # Fallback to "not spam"
