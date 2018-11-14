from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from azureml.train.automl.model_wrappers import LightGBMClassifier

manual_model = Pipeline(memory=None, steps=[('StandardScaler', StandardScaler(**scaler_params)), ('LightGBMClassifier', LightGBMClassifier(**model_params))])

manual_model.fit(X_train.values, y_train.values[:, 0])
y_pred = manual_model.predict(X_test)