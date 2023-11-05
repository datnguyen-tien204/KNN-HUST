import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np

df=pd.read_csv("weather.csv")
df_test=pd.read_csv("weather_test.csv")
hn_df = df[df['province'] == 'Ha Noi']

hn_df["wind_d"]=hn_df["wind_d"].astype('category')
df_test["wind_d"]=df_test["wind_d"].astype('category')

def encode_data(encoders, df, feature):
    encoder = LabelEncoder()
    encoded_data = encoder.fit_transform(df[feature])
    df[feature] = encoded_data
    encoders[feature] = encoder
    return encoders, df
def scale_data(scalers, df, feature):
    data = np.array(df[feature].tolist()).reshape(-1, 1)
    scaler = MinMaxScaler().fit(data)
    scaled_data = scaler.transform(data)
    df[feature] = scaled_data
    scalers[feature] = scaler
    return scalers, df

numeric_features = ['max', 'min', 'wind', 'humidi', 'cloud', 'pressure']
categorical_features = ['wind_d']
label = 'rain'

scalers = {}
scalers_test = {}
encoders = {}
encoders_test = {}

df_processed = hn_df.copy(deep=True)
df_test_processed = df_test.copy(deep=True)

for feature in numeric_features:
    scalers, df_processed = scale_data(scalers, df_processed, feature)
    scalers_test, df_test_processed = scale_data(scalers_test, df_test_processed, feature)

for feature in categorical_features:
    encoders, df_processed = encode_data(encoders, df_processed, feature)
    encoders_test, df_test_processed = encode_data(encoders_test, df_test_processed, feature)

print(df_processed.head())


def generator_dataset(dataframe):
    X_input = []
    y_input = []
    for i in range(len(dataframe)):
        x = []
        for feature in numeric_features:
            x.append(df_processed[feature].tolist()[i])
        for feature in categorical_features:
            x.extend([df_processed[feature].tolist()[i]])  # Sửa dòng này
        X_input.append(x)
        if df_processed[label].tolist()[i] > 0:
            y_input.append(1)
        else:
            y_input.append(0)
    X = np.array(X_input)
    y = np.array(y_input)
    return X, y
X_input,y_input=generator_dataset(df_processed)


X_test,y_test=generator_dataset(df_test_processed)


from sklearn.model_selection import train_test_split

X = np.array(X_input)
y = np.array(y_input)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=666)
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 11,weights='distance',p=1,metric='manhattan',algorithm='brute')  # n_neighbors means k
knn.fit(X_train, y_train)
prediction = knn.predict(X_val)
print("{} NN Score: {:.2f}%".format(11, knn.score(X_val, y_val)*100))


scoreList = []
for i in range(1,30):
    knn2 = KNeighborsClassifier(n_neighbors = i,weights='distance',p=1,metric='cityblock')  # n_neighbors means k
    knn2.fit(X_train, y_train)
    scoreList.append(knn2.score(X_val, y_val.T))

plt.plot(range(1,30), scoreList)
plt.xticks(np.arange(1,30,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

from sklearn.metrics import confusion_matrix
y_knn_pred=knn.predict(X_test)
knn_lr = confusion_matrix(y_test,y_knn_pred)


plt.figure(figsize=(24,12))

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)
plt.title("KNN Confusion Matrix",fontsize=24)
sns.heatmap(knn_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

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


if Tpmax > TPmin:
    btn = st.button("Predict")
    if btn:
        # Thực hiện dự đoán
        predicted_class = knn.predict(df_processed)
        if predicted_class[0] == 1:
            st.success("Today, Hanoi has rain.")
        else:
            st.success("Today, Hanoi didn't have rain.")
else:
    st.error("Maximum Temperature (TPmax) must be greater than Minimum Temperature (TPmin). Please adjust your inputs.")
