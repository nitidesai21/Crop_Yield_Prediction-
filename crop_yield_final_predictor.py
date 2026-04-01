# Generated from: crop_yield_final_predictor.ipynb
# Converted at: 2026-04-01T09:50:05.933Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import joblib
import pandas as pd
import numpy as np

model=joblib.load('crop_yield_random_search_cv_model.pkl')

encoder=joblib.load('all_label_encoders.pkl')

district_yield_lookup = joblib.load('district_avg_yield_map.pkl') 

def predict_crop_yield():
    print("------ Crop Yield Prediction System --------")
    
    # 2. Collect basic user inputs
    state = input("Enter state name: ").strip().title()
    district = input("Enter district name: ").strip().upper()
    
    try:
        year = int(input("Enter year (e.g., 2024): "))
    except ValueError:
        print("Error: Please enter a valid number for the year.")
        return
        
    # Handling your specific Season padding (.ljust)
    season = input("Enter season (eg: Kharif): ").strip().title().ljust(11)
    crop = input("Enter crop type (eg: Wheat): ").strip().title()
    
    try:
        area = float(input("Enter Area (in Hectares): "))
    except ValueError:
        print("Error: Please enter a valid number for the area.")
        return

    # 3. AUTOMATIC LOOKUP for District_Avg_Yield
    # This prevents the user from having to guess the average yield
    if district in district_yield_lookup:
        dist_avg = district_yield_lookup[district]
    else:
        # Fallback to a general average if the district is new/unseen
        dist_avg = district_yield_lookup.mean()

    # 4. Create DataFrame with your 7 features in the EXACT order
    user_data = pd.DataFrame([[
        state, district, year, season, crop, area, dist_avg
    ]], columns=['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 'Area', 'District_Avg_Yield'])

    # 5. Transform categorical variables to numbers
    try:
        user_data['State_Name'] = encoder['State_Name'].transform([state])[0]
        user_data['District_Name'] = encoder['District_Name'].transform([district])[0]
        user_data['Season'] = encoder['Season'].transform([season])[0]
        user_data['Crop'] = encoder['Crop'].transform([crop])[0]
    except KeyError as e:
        print(f"Error: {e} not found in training data. Check spelling/casing.")
        return

    # 6. Make prediction
    prediction = model.predict(user_data)
    
    print(f"\nPredicted Yield: {prediction[0]:.2f} tons per hectare")
    print(f"Estimated Total Production: {prediction[0] * area:.2f} tons")

# Run the system
predict_crop_yield()