import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import numpy as np

pd.set_option('display.max_columns', None)

# Load the top 10000 data
try:
    data = pd.read_csv('lol_top_10000.csv')

    # Drop unnecessary columns
    data = data.drop(['tier', 'more_url', 'rank', 'name'], axis=1)

    # Separate features (X) and target variable (y)
    X = data.drop('LP', axis=1)
    y = data['LP']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Define preprocessing steps for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('scaler', StandardScaler())  # Scale numerical features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
    ])

    # Drop rows with NaN values and reset index
    data = data.dropna()
    data = data.reset_index(drop=True)

    # scale the 'LP' column with min-max scaler, 0.8 ~ 1.0
    scaler = MinMaxScaler(feature_range=(0.8, 1.0))
    data['LP'] = scaler.fit_transform(data[['LP']])


except UnicodeDecodeError:
    print("Failed with encoding 'cp949'. Check your data encoding.")

# load champion data
champ_data = pd.read_csv('champ.csv')

# drop rows which column '챔피언' is NaN values
champ_data = champ_data.dropna(subset=['챔피언'])

# drop rows which column '주역할군' and '부역할군' are all NaN values
champ_data = champ_data.dropna(subset=['주역할군', '부역할군'], how='all')

# calculate win rate and add it to the dataframe
data['win_rate'] = data['wins'] / (data['wins'] + data['loses'])

# calculate each champion's win rate and add it to the dataframe
data['win_rate_champion0'] = data['champion_wins0'] / (data['champion_matches0'])
data['win_rate_champion1'] = data['champion_wins1'] / (data['champion_matches1'])
data['win_rate_champion2'] = data['champion_wins2'] / (data['champion_matches2'])
data['win_rate_champion3'] = data['champion_wins3'] / (data['champion_matches3'])
data['win_rate_champion4'] = data['champion_wins4'] / (data['champion_matches4'])

# make a new dataframe for user-based collaborative filtering matrix (user x champion)
# each row is user, each column is champion.
# column's name is champion's name. ('챔피언')
# initialize all values to 0
user_matrix = pd.DataFrame(0, index=data['Unnamed: 0'], columns=champ_data['챔피언'], dtype=np.float64)

# fill the user_matrix
# all user has 5 champions
# fill the user matrix with each user's 5 champions win rate x LP
for i in range(len(data)):
    # initialize a user's row to win rate x LP
    user_matrix.iloc[i] = 0.0
    for j in range(5):
        user_matrix.iloc[i][data.iloc[i][f'champion{j}']] = data.iloc[i][f'win_rate_champion{j}']

# save the user_matrix to csv file
user_matrix.to_csv('user_matrix_mod.csv')
