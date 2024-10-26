# importazone varie librerie
import pandas as pd
import numpy as np
import os

# librerie SKLearn
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, make_scorer

model_dir = os.environ['SM_MODEL_DIR']

houses_dataframe = pd.read_csv(model_dir+"/data/input/data.csv")
houses_dataframe.head()

prezzi_vendita = houses_dataframe["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(houses_dataframe, # x
                                                    prezzi_vendita, # y
                                                    test_size=0.2, # dimensione del test rispetto a train
                                                    random_state=12 # per riproducibilità
                                                    ) 

# verifica della divisione del dataset
print(len(X_train), len(X_test))

# selezione feature numeriche
feature_numeriche = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath"]

# ricreazione di train e test set con feature numeriche
X_train_numerical = X_train[feature_numeriche]
X_test_numerical = X_test[feature_numeriche]

# creazione modello
model = RandomForestRegressor(random_state=12)

# addestramento modello
model.fit(X_train_numerical, y_train)

# previsione
y_pred = model.predict(X_test_numerical)

mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"MAE: {mae}\nMAPE: {mape}")

# selezione feature categoriche
feature_categoriche = ["MSZoning", "Utilities", "Neighborhood", "SaleType", "SaleCondition"]

# ricreazione di train e test set con feature numeriche e cateogoriche selezionate
X_train_finale = X_train[feature_numeriche + feature_categoriche]
X_test_finale = X_test[feature_numeriche + feature_categoriche]

X_train_finale

# creazione one-hot encoder
onehot_encoder_column_transformer = ColumnTransformer(
    [("onehot_encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), feature_categoriche)], 
    remainder="passthrough", 
    verbose_feature_names_out=False
)

# creazione pipeline
pipeline_finale = Pipeline(
    [
        ("onehot_encoder", onehot_encoder_column_transformer),
        ("random_forest", RandomForestRegressor(random_state=12))
    ]
)

pipeline_finale.fit(X_train_finale, y_train)

# definizione del parametro della griglia di ricerca
params = {
    'random_forest__max_depth': [None, 10, 20]
}

gridsearch = GridSearchCV(
    estimator=pipeline_finale,
    param_grid=params,
    scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False),
    cv=KFold(n_splits=5),
    refit=True,
    verbose=2
)

# addestramento della grid search
gridsearch.fit(X_train_finale, y_train)

# salvataggio del modello
from sklearn.externals import joblib

joblib.dump(gridsearch, model_dir+'/model.joblib')