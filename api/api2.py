import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize FastAPI App
app = FastAPI(title="Stargazing Prediction API", 
              description="Predicts stargazing quality based on location and time",
              version="1.0.0")

# Add CORS middleware   
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define paths with environment variable fallbacks
weather_file_path = os.environ.get('WEATHER_DATA_PATH', 'GlobalWeatherRepository.csv')
model_dir = os.environ.get('MODEL_DIR', 'models')
model_path = os.path.join(model_dir, 'best_model.joblib')
scaler_path = os.path.join(model_dir, 'scaler.joblib')
features_path = os.path.join(model_dir, 'features.joblib')

# Load Weather Data (needed only for finding closest locations)
weather_data = pd.read_csv(weather_file_path)

# Load pre-trained models if available, otherwise train
print("Loading pre-trained models...")
best_model = joblib.load(model_path)
features = joblib.load(features_path)

# Scaler might not exist if best model is a tree-based model
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

print("Pre-trained models loaded successfully!")
print(f"Model features: {features}")

# Class for input validation
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
        day_of_year = int(input_datetime.strftime('%j'))  # Day of year (1-366)
        hour = input_datetime.hour
        is_night = 1 if (hour >= 18 or hour <= 5) else 0
        is_morning = 1 if (hour >= 6 and hour <= 11) else 0  # Added morning feature
        
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
            'is_night': is_night,
            'is_morning': is_morning  # Added morning feature
        }])
        
        # Verify all required features are present
        missing_features = [f for f in features if f not in input_features.columns]
        if missing_features:
            raise HTTPException(
                status_code=500, 
                detail=f"Missing features: {', '.join(missing_features)}"
            )
            
        # Scale if using linear regression
        if scaler is not None:
            input_scaled = scaler.transform(input_features[features])
            predicted_quality = best_model.predict(input_scaled)[0]
        else:
            predicted_quality = best_model.predict(input_features[features])[0]
        
        # Convert to percentage (0-100)
        stargazing_percentage = min(100, max(0, predicted_quality * 10))
        
        # Get nearest reference location
        nearest = closest_locations.iloc[0]
        
        # Determine time category for message
        time_category = "night" if is_night else "morning" if is_morning else "afternoon"
        
        return {
            "stargazing_quality_percentage": round(stargazing_percentage, 1),
            "reference_location": f"{nearest['country']}, {nearest['location_name']}",
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
                "is_night": bool(is_night),
                "is_morning": bool(is_morning),  # Added morning info
                "time_category": time_category  # Added time category
            },
            "message": f"The sky is estimated to be {stargazing_percentage:.1f}% clear for stargazing during {time_category} hours"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api2:app", host="0.0.0.0", port=port)