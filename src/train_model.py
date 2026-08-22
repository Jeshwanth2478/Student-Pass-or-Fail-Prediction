import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def create_datasets():
    data = {
        "study_hours": [
            1, 2, 2, 3, 3,
            4, 4, 5, 5, 6,
            6, 7, 7, 8, 8,
            9, 9, 10, 10, 11
        ],
        "attendance": [
            50, 55, 60, 62, 65,
            68, 70, 72, 75, 78,
            78, 80, 82, 85, 87,
            88, 90, 92, 94, 96
        ],
        "previous_score": [
            35, 40, 42, 45, 48,
            68, 70, 72, 75, 76,
            62, 65, 68, 70, 72,
            75, 78, 82, 85, 90
        ],
        "assignment_completed": [
            2, 2, 3, 3, 4,
            4, 5, 5, 6, 6,
            7, 7, 8, 8, 9,
            9, 10, 10, 10, 10
        ],
        "passed": [
            0, 0, 0, 0, 0,
            0, 0, 0, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1
        ]
    }

    return pd.DataFrame(data)

#create dataset
def train_model():
    df = create_datasets()
    print("\nDataset created successfully.")
    print("-"*50)
    print(df)
#define features and target 
    features = ["study_hours",
                "attendance",
                "previous_score",
                "assignment_completed"]
    X=df[features]
    Y=df["passed"]
#split the data training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                         Y,
                                                        test_size=0.2, 
                                                        random_state=42)
    print("\nTraining Samples:", len(X_train))
    print("Testing Samples:", len(X_test))
#Creating Ml model
    model = RandomForestClassifier(n_estimators=100, 
                                   random_state=42)
#Training the model
    print("\nTraining the model...")
    model.fit(X_train, Y_train)
    print("Model trained successfully.")
#Evaluating the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print(f"\nModel Accuracy:{accuracy:.2f}")
#save the trained model
    Model_path="model/model.pkl"
    joblib.dump(model, Model_path)
    print(f"\nModel saved successfully")
    print(f"Model path: {Model_path}")
if __name__ == "__main__":
    train_model()