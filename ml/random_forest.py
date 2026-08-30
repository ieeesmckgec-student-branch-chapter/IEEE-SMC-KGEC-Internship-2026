import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

# Read dataset
data = pd.read_csv("optimizer_data.csv")

# Features
X = data[[

    "tasks",

    "cpu",

    "memory",

    "bandwidth",

    "priority",

    "deadline"

]]

# Target
y = data["optimizer"]

# Split
X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)

# Random Forest
model = RandomForestClassifier(

    n_estimators=200,

    max_depth=10,

    random_state=42

)

model.fit(

    X_train,

    y_train

)

# Prediction
prediction = model.predict(

    X_test

)

# Accuracy
accuracy = accuracy_score(

    y_test,

    prediction

)

print("\nAccuracy:")

print(round(accuracy * 100, 2), "%")

print("\nClassification Report")

print(

    classification_report(

        y_test,

        prediction

    )

)

print("\nConfusion Matrix")

print(

    confusion_matrix(

        y_test,

        prediction

    )

)

# Feature Importance
importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nFeature Importance")

print(importance)

# Prediction Example
sample = pd.DataFrame({

    "tasks": [30000],

    "cpu": [60],

    "memory": [2048],

    "bandwidth": [500],

    "priority": [3],

    "deadline": [200]

})

print("\nPredicted Best Optimizer:")

print(model.predict(sample)[0])