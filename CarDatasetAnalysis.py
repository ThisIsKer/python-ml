import tensorflow as tf
import DatasetImport as di
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

file_path = di.getCarDatasetPath()
df = pd.read_csv(file_path)

print(df.head())

numerical_features = ["Year", "Engine_Size", "Mileage", "Doors", "Owner_Count"]
categorical_features = ["Brand", "Model", "Fuel_Type", "Transmission"]
target = "Price"

scaler = MinMaxScaler()

df = df.dropna() #Drop rows with null values

df[numerical_features] = scaler.fit_transform(df[numerical_features]) #Numerical data normalisation

df = pd.get_dummies(df, columns = categorical_features) #Dummification of categorical data

#Partition splitting
X = df.drop(columns=[target]) 
X = X.astype(float)  # Convert all boolean columns to float (0.0 and 1.0)
y = df[target] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Convert to array for TensorFlow
print(X_train.dtypes)
print(y_train.dtype)
X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = np.array(y_train), np.array(y_test)

# Define the model
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)  # Output layer for regression (price prediction)
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Model summary
model.summary()

# Model Training


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

# Model evaluation 
loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error: {mae:.2f}")

# Price prediction through regression

sample_data = np.expand_dims(X_test[0], axis=0)  # Select one sample
predicted_prices = model.predict(X_test[:10])  # Predict for the first 10 test samples
actual_prices = y_test[:10]  # Get the first 10 actual prices

for i in range(10):
    print(f"Car {i+1}: Actual Price = ${actual_prices[i]:.2f}, Predicted Price = ${predicted_prices[i][0]:.2f}")


# Standardize the features before applying PCA (important for PCA performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensionality (let’s try reducing to 10 components)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Print explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by each component: {explained_variance}")
print(f"Total variance explained: {sum(explained_variance):.2f}")

# Assuming 'df' is your original DataFrame before applying PCA
feature_names = df.columns  # Get original feature names

# Create a DataFrame showing how much each feature contributes to each principal component
pca_components_df = pd.DataFrame(pca.components_, columns=feature_names, index=[f"PC{i+1}" for i in range(pca.n_components)])

# Display the DataFrame
print(pca_components_df)