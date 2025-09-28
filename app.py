import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import os
import joblib  # For saving and loading models

# Create the Flask app instance
app = Flask(__name__)

# Function to load the dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    # Use 'ISO8601' format which handles "YYYY-MM-DD"
    df['Date'] = pd.to_datetime(df['Date'], format=None)  # Let pandas infer the format (ISO8601)
    df.set_index('Date', inplace=True)
    return df


# Preprocess data (select relevant columns, handle missing values, normalize)
def preprocess_data(df, column='Amazon_Price', lookback=60):
    # Drop rows with missing values
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")
    df = df.dropna(subset=[column])

    # Select the stock price column for prediction
    stock_data = df[column].values
    stock_data = stock_data.reshape(-1, 1)

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data_scaled = scaler.fit_transform(stock_data)

    # Prepare the data for training the model (using 60 previous days to predict the next day)
    X, y = [], []
    for i in range(lookback, len(stock_data_scaled)):
        X.append(stock_data_scaled[i-lookback:i, 0])
        y.append(stock_data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    
    # Reshape X to be compatible with the model input
    X = np.reshape(X, (X.shape[0], X.shape[1]))

    return X, y, scaler

# Function to build and train the Linear Regression model
def build_and_train_model(X_train, y_train):
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Save the model using joblib
    joblib.dump(model, 'trained_model.pkl')

    return model

# Function to make predictions using the trained model
def make_predictions(model, X_test, scaler):
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1, 1))
    return predicted_stock_price

# Function to prepare the input data for prediction (for the next 30 days)
def prepare_for_prediction(df, column='Amazon_Price', lookback=60, scaler=None):
    # Get the last 'lookback' days of stock prices for prediction
    last_60_days = df[column].values[-lookback:].reshape(-1, 1)

    # Use the provided scaler to normalize the data
    last_60_days_scaled = scaler.transform(last_60_days)

    # Reshape data to match the model input
    X_input = np.reshape(last_60_days_scaled, (1, last_60_days_scaled.shape[0]))
    return X_input

# Function to load the saved model
def load_trained_model():
    if os.path.exists('trained_model.pkl'):
        model = joblib.load('trained_model.pkl')
        return model
    else:
        print("Model not found. Please train the model first.")
        return None

# Main function to predict the next 30 days of stock prices
def predict_next_30_days(df, column='Amazon_Price'):
    # Load the scaler used during training (this is important for the correct scaling during prediction)
    _, _, scaler = preprocess_data(df, column)

    # Prepare data for prediction
    X_input = prepare_for_prediction(df, column, scaler=scaler)

    # Load the trained model
    model = load_trained_model()

    if model is None:
        return None

    # Predict the next 30 days
    predictions = []
    for _ in range(30):
        predicted_price = model.predict(X_input)
        predicted_price = predicted_price[0]

        # Append the prediction to the predictions list
        predictions.append(predicted_price)

        # Update the input for the next prediction (adjusted for linear regression)
        X_input = np.reshape(np.append(X_input[0, 1:], predicted_price), (1, -1))

    # Convert the predictions back to the original price scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return predictions


# Home page route
@app.route('/')
def index():
    return render_template('index.html')  # This will render the homepage (index.html)

# Stock Prediction Page route
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    predictions = None
    labels = []
    prices = []

    if request.method == 'POST':
        # Get selected stock/commodity (in this case, 'Amazon_Price')
        selected_stock = request.form['stock']

        # Load dataset and make predictions
        df = load_data('Stock Market Dataset.csv')

        try:
            predictions = predict_next_30_days(df, column=selected_stock)
        except ValueError as e:
            return str(e)  # Display error message if column doesn't exist

        if predictions is not None:
            labels = [f"Day {i+1}" for i in range(30)]  # Create labels for the next 30 days
            prices = [float(pred) for pred in predictions]  # Convert predictions to float

    return render_template('prediction.html', predictions=predictions, labels=labels, prices=prices)

# Route for Data Analysis Page
@app.route('/analysis', methods=['GET'])
def analysis():
    # Load your data (or return relevant analysis results)
    df = load_data('Stock Market Dataset.csv')  # Or your data source
    return render_template('analysis.html', stock_data=df)

# Route for About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for Contact Page
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Here, you can process the contact form (e.g., send an email or save data)
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        print(f"Contact form submitted by {name} ({email}): {message}")
        return redirect(url_for('index'))  # Redirect back to home after form submission
    return render_template('contact.html')  # This will render the contact form (contact.html)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
