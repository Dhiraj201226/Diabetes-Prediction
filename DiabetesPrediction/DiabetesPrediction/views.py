from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):

    # Load dataset
    data = pd.read_csv("C:/Users/dhira/Downloads/diabetes.csv")

    X = data.drop("Outcome", axis=1)
    Y = data["Outcome"]

    # Split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.18, random_state=42
    )

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    # Get user input safely
    try:
        val1 = float(request.GET.get('n1', 0))
        val2 = float(request.GET.get('n2', 0))
        val3 = float(request.GET.get('n3', 0))
        val4 = float(request.GET.get('n4', 0))
        val5 = float(request.GET.get('n5', 0))
        val6 = float(request.GET.get('n6', 0))
        val7 = float(request.GET.get('n7', 0))
        val8 = float(request.GET.get('n8', 0))
    except ValueError:
        return render(request, 'predict.html', {
            "result2": "Please enter valid numeric values."
        })

    # Prepare and scale user input
    input_data = [[val1, val2, val3, val4, val5, val6, val7, val8]]
    input_data = scaler.transform(input_data)

    # Predict
    pred = model.predict(input_data)

    # Convert prediction to readable output
    if pred[0] == 1:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render(request, 'predict.html', {"result2": result1})
