import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    features = df[['amount', 'hour', 'frequency', 'location_change']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, features.columns
