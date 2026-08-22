#creating the fastapi application
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app=FastAPI(title="Student Performance Prediction API",
             description="This API predicts whether a student will pass or fail based on their study hours, attendance, previous score, and assignments completed.", 
             version="1.0.0")
#loading the model
model = joblib.load(Path(__file__).parent.parent / "model" / "model.pkl")
#defining the input data structure
class StudentData(BaseModel):
    study_hours: float
    attendance: float
    previous_score: float
    assignment_completed: int
#home endpoint
@app.get("/")
def home():
    return {"message": "Student Performance Prediction API is running."}
#health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}
#prediction endpoint
@app.post("/predict")
def predict_student(student: StudentData):
    data=pd.DataFrame({
        "study_hours": [student.study_hours],
        "attendance": [student.attendance],
        "previous_score": [student.previous_score],
        "assignment_completed": [student.assignment_completed]
    })
    prediction = model.predict(data)
    probability = model.predict_proba(data)
    if prediction[0] == 1:
        result = 'Pass'
    else:
        result = 'Fail'
    return {
        "prediction": result,
        "pass_probability":float(probability[0,1])
    }