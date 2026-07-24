from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pandas as pd
import joblib

# Create FastAPI App
app = FastAPI()

# Load Model and Preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Templates
templates = Jinja2Templates(directory="templates")

# Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ===========================
# Home Page
# ===========================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )


# ===========================
# Prediction Route
# ===========================

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    longitude: float = Form(...),
    latitude: float = Form(...),
    housing_median_age: float = Form(...),
    total_rooms: float = Form(...),
    total_bedrooms: float = Form(...),
    population: float = Form(...),
    households: float = Form(...),
    median_income: float = Form(...),
    ocean_proximity: str = Form(...)
):

    # Create DataFrame
    data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    # Preprocess Data
    prepared_data = preprocessor.transform(data)

    # Predict
    prediction = model.predict(prepared_data)

    # Return Result
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "prediction": round(prediction[0], 2)
        }
    )