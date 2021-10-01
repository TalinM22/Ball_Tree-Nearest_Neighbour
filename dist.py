import numpy as np
from sklearn.neighbors import BallTree
import pandas as pd

## Dummy dataframe A
df_A = pd.DataFrame(
    {
        "City": ["Buenos Aires", "Brasilia", "Santiago", "Bogota", "Caracas"],
        "Latitude": [-34.58, -15.78, -33.45, 4.60, 10.48],
        "Longitude": [-58.66, -47.91, -70.66, -74.08, -66.86],
    }
)

## Dummy dataframe B
df_B = pd.DataFrame(
    {
        "City": ["Onslow", "Newman", "Mullewa", "Kununurra", "Carnarvon"],
        "Latitude": [-21.6689, -23.4169, -28.5367, -15.7814, -24.8878],
        "Longitude": [115.10926, 119.7989, 115.5142, 128.71, 113.67],
    }
)

## Converting latitude and longitude to radians using numpy function deg2rad
for column in df_A[["Latitude", "Longitude"]]:
    rad = np.deg2rad(df_A[column].values)
    df_A[f"{column}_rad"] = rad

for column in df_B[["Latitude", "Longitude"]]:
    rad = np.deg2rad(df_B[column].values)
    df_B[f"{column}_rad"] = rad

## Constructing the KDTree using latitude-longitude values from dataframe A
ball = BallTree(df_A[["Latitude_rad", "Longitude_rad"]].values, metric="haversine")

## Number of neighbours to return
k = 1

## Results in two arrays: distance and index of neighbouring location 
df_B["Distance"], df_B["id_nearest"] = ball.query(
    df_B[["Latitude_rad", "Longitude_rad"]].values,
    k=k,
)

## Maps te ID to City name
df_B["City_nearest"] = df_B["id_nearest"].map(df_A["City"])

## Multiplying distance column with 6371(radius of Earth)
df_B["Distance"] = df_B["Distance"].astype(float).multiply(6371)

## Returns a filtered Dataframe B
df_B = df_B[["City", "Latitude", "Longitude", "City_nearest", "Distance"]]

print(df_B)
