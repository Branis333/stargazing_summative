from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Initialize FastAPI App
app = FastAPI(
    title="Stargazing Prediction API",
    description="Predicts stargazing quality based on location and time",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and dataset
MODEL_PATH = r'C:\Users\user\Desktop\stargazing_summative\linear_regression\stargazing_model.pkl'
DATA_PATH = r'C:\Users\user\Desktop\stargazing_summative\linear_regression\GlobalWeatherRepository.csv'

try:
    model = joblib.load(MODEL_PATH)  # Load trained model
    weather_data = pd.read_csv(DATA_PATH)  # Load dataset
    print("Model and data loaded successfully.")
except Exception as e:
    print(f"Error loading model or data: {e}")
    model, weather_data = None, None

# Updated features list to include time features
FEATURES = [
    'latitude', 'longitude', 'cloud', 'humidity', 
    'air_quality_PM2.5', 'air_quality_PM10', 'visibility_km', 'uv_index',
    'month', 'day_of_year', 'hour', 'is_night'  # Added time features
]

# Input validation model
class PredictionInput(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (-180 to 180)")
    datetime: str = Field(..., description="Date and time (YYYY-MM-DD HH:MM:SS)")

@app.post("/predict/")
def predict_stargazing_quality(data: PredictionInput):
    """
    Predict stargazing quality for a given location and time
    
    Returns a percentage (0-100) indicating how clear the sky is expected to be for stargazing
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")
        
    try:
        # Parse datetime
        try:
            input_datetime = datetime.strptime(data.datetime, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid datetime format. Use YYYY-MM-DD HH:MM:SS")
            
        # Find most similar locations in dataset
        weather_data['distance'] = np.sqrt(
            (weather_data['latitude'] - data.latitude)**2 + 
            (weather_data['longitude'] - data.longitude)**2
        )
        
        # Get 3 closest locations
        closest_locations = weather_data.nsmallest(3, 'distance')
        
        # Extract time-related features
        month = input_datetime.month
        hour = input_datetime.hour
        day_of_year = int(input_datetime.strftime('%j'))  # Day of year (1-366)
        is_night = 1 if (hour >= 18 or hour <= 5) else 0
        
        # Calculate weighted average of features based on distance
        weights = 1 / (closest_locations['distance'] + 0.01)  # Avoid division by zero
        weights = weights / weights.sum()
        
        # Create feature vector for prediction with all required features
        input_features = pd.DataFrame([{
            'latitude': data.latitude,
            'longitude': data.longitude,
            'cloud': (closest_locations['cloud'] * weights).sum(),
            'humidity': (closest_locations['humidity'] * weights).sum(),
            'air_quality_PM2.5': (closest_locations['air_quality_PM2.5'] * weights).sum(),
            'air_quality_PM10': (closest_locations['air_quality_PM10'] * weights).sum(),
            'visibility_km': (closest_locations['visibility_km'] * weights).sum(),
            'uv_index': (closest_locations['uv_index'] * weights).sum() * (1 - is_night),  # UV is 0 at night
            # Add time features
            'month': month,
            'day_of_year': day_of_year,
            'hour': hour,
            'is_night': is_night
        }])
        
        # Make prediction using the model
        predicted_quality = model.predict(input_features[FEATURES])[0]
        
        # Convert to percentage (0-100)
        stargazing_percentage = min(100, max(0, predicted_quality * 10))
        
        # Get nearest reference location
        nearest = closest_locations.iloc[0]
        
        # Prepare location reference - handle if columns don't exist
        location_ref = ""
        if 'country' in nearest and 'location_name' in nearest:
            location_ref = f"{nearest['country']}, {nearest['location_name']}"
        elif 'country' in nearest:
            location_ref = nearest['country']
        elif 'location_name' in nearest:
            location_ref = nearest['location_name']
        else:
            location_ref = f"Nearest point ({nearest['latitude']:.2f}, {nearest['longitude']:.2f})"
        
        return {
            "stargazing_quality_percentage": round(stargazing_percentage, 1),
            "reference_location": location_ref,
            "predicted_conditions": {
                "cloud_cover": round(input_features['cloud'].values[0], 1),
                "humidity": round(input_features['humidity'].values[0], 1),
                "air_quality_PM2.5": round(input_features['air_quality_PM2.5'].values[0], 1),
                "air_quality_PM10": round(input_features['air_quality_PM10'].values[0], 1),
                "visibility_km": round(input_features['visibility_km'].values[0], 1)
            },
            "is_night": bool(is_night),
            "time_info": {
                "month": month,
                "day_of_year": day_of_year,
                "hour": hour,
                "is_night": bool(is_night)
            },
            "message": f"The sky is estimated to be {stargazing_percentage:.1f}% clear for stargazing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
