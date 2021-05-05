import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib

scaled_features = pd.read_csv("TRAIN.csv")
y = scaled_features['Price']
X = scaled_features.drop(['Price', 'Cars'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=25)

regressor1 = xgb.XGBRegressor(n_estimators=200, gamma=0, max_depth=4)

regressor1.fit(X_train, y_train)
y_Pred = regressor1.predict(X_test)


filename = 'finalizedmodel.sav'
joblib.dump(regressor1, filename)
