import joblib
import pandas as pd
def load_model():
    model = joblib.load("model/model.pkl")
    print("model loaded succesfully")
    return model
def predict_student(model):
    Student=pd.DataFrame({
        "study_hours": [7],
        "attendence": [85],
        "previous_score": [75],
        "assignment_completed": [8]
    })
    prediction = model.predict(Student)
    probability = model.predict_proba(Student)
    return prediction[0], probability[0,1]
def main():
    model = load_model()
    prediction, probability = predict_student(model)
    if prediction == 1:
        result = 'Pass'
    else:
        result = 'Fail'
    print("\nStudent Prediction")
    print("-"*30)
    print(f"prediction:{result}")
    print(f"pass Probability:{probability:.2f}")
if __name__ == "__main__":
    main()