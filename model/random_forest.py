import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("D:\\Teng He\\Desktop\\COVID-19-Severity-Prediction-App\\data\\Covid-19 Cleaned Data.csv")

# assigning values to features as X and target as Y
X = data.drop(["hasil dignosis"], axis=1)  # Features
y = data["hasil dignosis"]  # Target variable

scaler = StandardScaler()
X = scaler.fit_transform(X)

# splitting test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Create and train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model as a .pkl file
with open("D:\\Teng He\\Desktop\\COVID-19-Severity-Prediction-App\\model\\random_forest_model.pkl", "wb") as file:
    pickle.dump(model, file)
