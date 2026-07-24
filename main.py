import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Load The DataSet
housing=pd.read_csv("housing.csv")

# 2 Create a test data
train_set, test_set = train_test_split(
    housing,
    test_size=0.2,
    random_state=42
)

data=train_set.copy()
house_price=data["median_house_value"].copy()
df=data.drop("median_house_value",axis=1)

# 3 Separate Numerical and categorical columns
num_attribs=df.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribs=["ocean_proximity"]

# 4 making pipeline for numerical columns
num_pipline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler()),
])

# 5 making pipeline for categorical 
cat_pipline=Pipeline([
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

# 6 construct the full pipeline 
full_pipeline=ColumnTransformer([
    ("num",num_pipline,num_attribs),
    ("cat",cat_pipline,cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(df)

housing_prepared = pd.DataFrame(
    housing_prepared,
    columns=full_pipeline.get_feature_names_out()
)

# Random Forest
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, house_price)
forest_preds = forest_reg.predict(housing_prepared)

# joblib.dump(forest_reg, "model.pkl")
# joblib.dump(full_pipeline, "preprocessor.pkl")

joblib.dump(forest_reg, "model.pkl", compress=3)
joblib.dump(full_pipeline, "preprocessor.pkl", compress=3)