# Stargazing Summative

## Project Architecture and Workflow

This project uses a multi-stage workflow for development, testing, and deployment:

### Development and Model Training

- **Jupyter Notebook (`linear_regression/stargazing_model.ipynb`)**: Used for exploratory data analysis, feature engineering, and initial model development. The notebook trains three different models (Linear Regression, Decision Tree, and Random Forest) and evaluates their performance against the dataset. This is where we developed the stargazing quality formula and identified key features for prediction.

- **Training Script (`api/train_and_save_models.py`)**: For deployment, we use this dedicated script that creates optimized models with smaller file sizes. The script:
  - Loads and preprocesses the global weather dataset
  - Trains multiple model types with optimized hyperparameters
  - Selects the best performing model
  - Saves the model, scaler (if applicable), and features list to the `models/` directory
  - Tests time sensitivity to verify the models account for different times of day

### API Implementation

- **Testing API (`api/api.py`)**: Used during development for local testing. This version:
  - Uses absolute file paths specific to the development environment
  - Loads the model saved directly from the notebook
  - Good for iterative development but not suitable for deployment

- **Production API (`api/api2.py`)**: The deployment-ready version that:
  - Uses environment variables and relative paths for better portability
  - Has more robust error handling
  - Is designed to work with the models saved by `train_and_save_models.py`
  - Includes additional features like time category classification
  - Deployed to a cloud service (Render) for public access

### Model Features

The model predicts stargazing quality based on:
- Location factors (latitude, longitude)
- Weather conditions (cloud cover, humidity, visibility)
- Air quality (PM2.5, PM10)
- Time factors (month, day of year, hour)
- Time classifications (is_night, is_morning)

The final model applies bonuses for nighttime viewing and penalties for morning viewing, aligning with optimal stargazing conditions.

## Public API Endpoint
The API endpoint for predictions is available at:
[Stargazing API on Render](https://stargazing-hrqk.onrender.com/docs#/)

## YouTube Video Demo
Watch the video demo here:
[YouTube Video Demo](https://drive.google.com/file/d/1m69cI2Nyp7s2tSjblaXsBhEOSs604ny5/view?usp=sharing)

## Running the Mobile App
To run the mobile app, follow these steps:

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/Branis333/stargazing_summative
    cd client/stargazing_app/ 
    ```

2. **Install Dependencies**:
    ```sh
    flutter pub get
    ```

3. **Run the App**:
    ```sh
    flutter run
    ```

Make sure you have Flutter installed and set up on your machine. For more information, visit the [Flutter installation guide](https://flutter.dev/docs/get-started/install).