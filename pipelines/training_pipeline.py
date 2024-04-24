from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
# chache is enabled coz if nothing changes in the data, the old data is used
@pipeline(enable_cache=True)
def train_pipeline(data_path:str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2, rmse = evaluate_model(model, X_test, y_test)
    train_model(df)
    evaluate_model(df)
