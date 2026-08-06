import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
TEST_SIZE = 0.2

def preprocess_data(input_file='SeoulBikeData.csv',
                    train_output='processed_bike_data_train.csv',
                    test_output='processed_bike_data_test.csv'):
    
    # 1. Load Data
    df = pd.read_csv(input_file, encoding='unicode_escape')

    # 2. Handle Missing Values
    df = df.dropna()

    # 3. Rich Temporal Feature Engineering
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # Cyclical Encoding for Hour and Month (Helps trees understand 23 wraps to 0)
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    df = df.drop(columns=['Date'])

    # 4. Encode Categorical Variables
    df = pd.get_dummies(df, columns=['Seasons'], prefix='Season')

    binary_cols = ['Holiday', 'Functioning Day']
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # 5. Split Data (No target clipping, no scaling)
    X = df.drop(columns=['Rented Bike Count'])
    y = df['Rented Bike Count']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 6. Save train and test separately
    train_df = X_train.copy()
    train_df['Rented Bike Count'] = y_train
    test_df = X_test.copy()
    test_df['Rented Bike Count'] = y_test

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    print(f"Preprocessing complete.")
    print(f"Train saved to {train_output} ({len(train_df)} rows).")
    print(f"Test saved to {test_output} ({len(test_df)} rows).")

    return train_df, test_df

if __name__ == "__main__":
    preprocess_data()