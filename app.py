from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from typing import Dict

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the model
model = load_model('my_model_nn_1.h5')

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("full.html", {"request": request})

@app.post("/predict")
async def predict(
    gender: str = Form(...),
    age: float = Form(...),
    height: float = Form(...),
    weight: float = Form(...),
    family_history_with_overweight: str = Form(...),
    favc: str = Form(...),
    fcvc: float = Form(...),
    ncp: float = Form(...),
    caec: str = Form(...),
    smoke: str = Form(...),
    ch2o: float = Form(...),
    scc: str = Form(...),
    faf: float = Form(...),
    tue: float = Form(...),
    calc: str = Form(...),
    mtrans: str = Form(...)
):
    try:
        # Create a dictionary from the form data
        data = {
            'gender': gender,
            'age': age,
            'height': height,
            'weight': weight,
            'family_history_with_overweight': family_history_with_overweight,
            'favc': favc,
            'fcvc': fcvc,
            'ncp': ncp,
            'caec': caec,
            'smoke': smoke,
            'ch2o': ch2o,
            'scc': scc,
            'faf': faf,
            'tue': tue,
            'calc': calc,
            'mtrans': mtrans
        }
        
        # Convert to DataFrame and preprocess
        data_df = pd.DataFrame([data])
        processed_data = preprocess(data_df)
        
        # Get prediction
        prediction = get_prediction(processed_data, model)
        
        return {"prediction": prediction[0]}
    except Exception as e:
        return {"error": str(e)}

def preprocess(data):
    # Map categorical variables to numerical values
    mappings = {
        'gender': {'Male': 0, 'Female': 1},
        'mtrans': {
            'Automobile': 0, 'Bike': 1, 'Motorbike': 2, 
            'Public_Transportation': 3, 'Walking': 4
        },
        'family_history_with_overweight': {'no': 0, 'yes': 1},
        'favc': {'no': 0, 'yes': 1},
        'smoke': {'no': 0, 'yes': 1},
        'scc': {'no': 0, 'yes': 1},
        'caec': {
            'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3
        },
        'calc': {
            'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3
        }
    }

    # Apply mappings to the data
    for column, mapping in mappings.items():
        if column in data:
            data[column] = data[column].map(mapping).astype(float)

    # Convert numerical columns to float
    numerical_cols = ['age', 'height', 'weight', 'fcvc', 'ncp', 'faf', 'tue', 'ch2o']
    for col in numerical_cols:
        if col in data:
            data[col] = data[col].astype(float)

    return data

def get_prediction(data, model):
    predictions = model.predict(data)
    highest_labels = np.argmax(predictions, axis=1)
    map_label = {
        0: 'Insufficient Weight', 1: 'Normal Weight', 2: 'Overweight Level I',
        3: 'Overweight Level II', 4: 'Obesity Type I', 5: 'Obesity Type II', 6: 'Obesity Type III'
    }
    highest_labels_mapped = [map_label[label] for label in highest_labels]
    return highest_labels_mapped

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
