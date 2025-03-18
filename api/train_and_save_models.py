import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Load Weather Data
weather_file_path = 'GlobalWeatherRepository.csv'  # Update path if needed
weather_data = pd.read_csv(weather_file_path)

# Prepare data for model training
def prepare_data():
    # Extract datetime features from last_updated
    print("Extracting time features from last_updated...")
    weather_data['datetime'] = pd.to_datetime(weather_data['last_updated'])
    weather_data['month'] = weather_data['datetime'].dt.month
    weather_data['hour'] = weather_data['datetime'].dt.hour
    weather_data['day_of_year'] = weather_data['datetime'].dt.dayofyear
    weather_data['is_night'] = ((weather_data['hour'] >= 18) | 
                              (weather_data['hour'] <= 5)).astype(int)
    
    # Fill missing values
    weather_data_clean = weather_data.fillna({
        'cloud': weather_data['cloud'].mean(),
        'humidity': weather_data['humidity'].mean(),
        'air_quality_PM2.5': weather_data['air_quality_PM2.5'].mean(),
        'air_quality_PM10': weather_data['air_quality_PM10'].mean(),
        'visibility_km': weather_data['visibility_km'].mean(),
        'uv_index': weather_data['uv_index'].mean()
    })
    
    # Create stargazing quality score (0-10 scale, higher is better)
    weather_data_clean['stargazing_quality'] = (
        (100 - weather_data_clean['cloud']) * 0.35 +
        (100 - weather_data_clean['humidity']) * 0.25 +
        (100 - np.clip(weather_data_clean['air_quality_PM2.5'] * 2, 0, 100)) * 0.15 +
        (100 - np.clip(weather_data_clean['air_quality_PM10'], 0, 100)) * 0.15 +
        # Add bonus points for nighttime (5% boost)
        (weather_data_clean['is_night'] * 5)
    ) / 100 * 10
    
    return weather_data_clean

# Train models and save them
def train_and_save_models(use_small_model=True):
    print("Training and saving models...")
    data = prepare_data()
    
    # Select features and target - now including time features
    features = [
        # Original features
        'latitude', 'longitude', 'cloud', 'humidity', 'air_quality_PM2.5', 
        'air_quality_PM10', 'visibility_km', 'uv_index',
        # New time features
        'month', 'day_of_year', 'hour', 'is_night'
    ]
    
    # Verify all features exist
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"WARNING: Missing features: {missing_features}")
        features = [f for f in features if f in data.columns]
        
    X = data[features]
    y = data['stargazing_quality']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    lr_mse = mean_squared_error(y_test, lr_preds)
    lr_r2 = r2_score(y_test, lr_preds)
    print(f"Linear Regression - MSE: {lr_mse:.4f}, R²: {lr_r2:.4f}")
    
    # Train Decision Tree (reduced complexity)
    dt_model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=4, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    dt_mse = mean_squared_error(y_test, dt_preds)
    dt_r2 = r2_score(y_test, dt_preds)
    print(f"Decision Tree - MSE: {dt_mse:.4f}, R²: {dt_r2:.4f}")
    
    # Train Random Forest (adjust size based on parameter)
    if use_small_model:
        # Smaller model for GitHub compatibility
        print("Creating smaller RandomForest model for easier deployment...")
        rf_model = RandomForestRegressor(
            n_estimators=20,      # Reduced from 100 
            max_depth=10,         # Limited depth
            min_samples_leaf=4,   # Require more samples per leaf
            random_state=42
        )
    else:
        # Full model (original size, might be too large for GitHub)
        print("Creating full-sized RandomForest model (larger file size)...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    print(f"Random Forest - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")
    
    # Find best model based on MSE
    models = {
        'linear_regression': (lr_model, lr_mse, scaler),
        'decision_tree': (dt_model, dt_mse, None),
        'random_forest': (rf_model, rf_mse, None)
    }
    
    best_model_name = min(models, key=lambda k: models[k][1])
    best_model, _, best_scaler = models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save with compression for smaller file size
    print("Saving models with compression...")
    joblib.dump(best_model, 'models/best_model.joblib', compress=9)  # Max compression
    
    if best_scaler is not None:
        joblib.dump(best_scaler, 'models/scaler.joblib', compress=9)
    
    # Save features list for reference
    joblib.dump(features, 'models/features.joblib', compress=9)
    print("Features saved:", features)
    
    # Also save features as text for easy reference
    with open('models/features.txt', 'w') as f:
        f.write(','.join(features))
    
    # Show model size
    model_size = os.path.getsize('models/best_model.joblib') / (1024 * 1024)  # Size in MB
    print(f"Model file size: {model_size:.2f} MB")
    
    print("Models saved successfully!")
    return best_model_name, rf_r2

# Test date sensitivity
def test_date_sensitivity(model, features, scaler=None):
    print("\nTesting date sensitivity...")
    test_data = [{
        'latitude': 40.7, 'longitude': -74.0,  # NYC
        'cloud': 20, 'humidity': 60, 'air_quality_PM2.5': 15, 'air_quality_PM10': 20,
        'visibility_km': 10, 'uv_index': 5,
        'month': 5, 'day_of_year': 135, 'hour': 12, 'is_night': 0  # Daytime
    }, {
        'latitude': 40.7, 'longitude': -74.0,  # NYC
        'cloud': 20, 'humidity': 60, 'air_quality_PM2.5': 15, 'air_quality_PM10': 20,
        'visibility_km': 10, 'uv_index': 0,
        'month': 5, 'day_of_year': 135, 'hour': 22, 'is_night': 1  # Nighttime
    }]
    
    test_df = pd.DataFrame(test_data)
    
    # Make predictions
    if scaler:
        test_scaled = scaler.transform(test_df[features])
        predictions = model.predict(test_scaled)
    else:
        predictions = model.predict(test_df[features])
        
    print(f"Daytime quality: {predictions[0]:.2f}/10")
    print(f"Nighttime quality: {predictions[1]:.2f}/10")
    print(f"Difference: {predictions[1] - predictions[0]:.2f}")

if __name__ == "__main__":
    # Choose which model size to create (True for small, False for full-size)
    use_small = True
    
    # Train and save the models
    best_model_name, rf_accuracy = train_and_save_models(use_small_model=use_small)
    
    # Check if the model is date sensitive
    best_model = joblib.load('models/best_model.joblib')
    features = joblib.load('models/features.joblib')
    scaler = None
    if best_model_name == 'linear_regression':
        scaler = joblib.load('models/scaler.joblib')
    test_date_sensitivity(best_model, features, scaler)
    
    print("\n========== SUMMARY ==========")
    print(f"Best model type: {best_model_name}")
    print(f"RandomForest R² score: {rf_accuracy:.4f}")
    print(f"Model size optimized for GitHub: {'Yes' if use_small else 'No'}")
    print(f"Features used: {', '.join(features)}")
    print("==============================")