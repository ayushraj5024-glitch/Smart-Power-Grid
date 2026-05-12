import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from utils import air_density, wind_power

df = pd.read_csv("../dataset/WFT.csv")

le = LabelEncoder()

df['Rain_encoded'] = le.fit_transform(df['Rain'])

df['Air_Density'] = df.apply(
    lambda r: air_density(
        r['Temperature'],
        r['Humidity'],
        r['Pressure']
    ),
    axis=1
)

df['Power'] = df.apply(
    lambda r: wind_power(
        r['Wind_Speed'],
        r['Air_Density']
    ),
    axis=1
)

X = df[['Temperature','Humidity','Wind_Speed','Cloud_Cover','Pressure','Rain_encoded']]

y = df['Power']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()

model.fit(X_train, y_train)

joblib.dump(model, "../models/model.pkl")

print("Model Saved")