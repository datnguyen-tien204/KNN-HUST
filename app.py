import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import pickle
import subprocess


st.header("Predict Hanoi Weather")
lst_wind_d = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
wind_directions = [
    "North", "North-Northeast", "Northeast", "East-Northeast",
    "East", "East-Southeast", "Southeast", "South-Southeast",
    "South", "South-Southwest", "Southwest", "West-Southwest",
    "West", "West-Northwest", "Northwest", "North-Northwest"]
def choose(choose):
    if choose in wind_directions:
        index = wind_directions.index(choose)
        return lst_wind_d[index]
    else:
        return None

TPmin = st.slider("Minimum Temperatures(Celsius)", 0, 100, 20)
Tpmax = st.slider("Maximum Temperatures(Celsius)", 0, 100, 30)
Wd = st.slider("Wind Speed(km/h)", 0, 150, 10)
wind_d = st.selectbox("Wind Direction", wind_directions)
Humidity = st.slider("Humidity(%)", 0, 100, 80)
Cloud = st.slider("Cloud(%)", 0, 100, 10)
pressure=st.slider("Pressure(Bar)", 900, 1050, 1000)

data_list = [[Tpmax, TPmin, Wd, wind_d, Humidity, Cloud, pressure]]

# Tạo DataFrame từ danh sách dữ liệu
df = pd.DataFrame(data_list, columns=["max", "min", "wind", "wind_d", "humidi", "cloud", "pressure"])
df["wind_d"] = df["wind_d"].astype('category')

def encode_data(encoders, df, feature):
    encoder = LabelEncoder()
    df[feature] = encoder.fit_transform(df[feature])
    encoders[feature] = encoder
    return encoders, df

numeric_features = ['max', 'min', 'wind', 'humidi', 'cloud', 'pressure']
categorical_features = ['wind_d']
label = 'rain'

scalers = {}
encoders = {}

df_processed = df.copy(deep=True)
for feature in categorical_features:
    encoders, df_processed = encode_data(encoders, df_processed, feature)

with open('knn_model.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

if Tpmax > TPmin:
    btn = st.button("Predict")
    if btn:
        # Thực hiện dự đoán
        predicted_class = knn_model.predict(df_processed)
        if predicted_class[0] == 1:
            st.success("Today, Hanoi has rain.")
        else:
            st.success("Today, Hanoi didn't have rain.")
else:
    st.error("Maximum Temperature (TPmax) must be greater than Minimum Temperature (TPmin). Please adjust your inputs.")
