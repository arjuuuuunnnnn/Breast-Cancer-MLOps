from zenml import pipeline
from steps.ingest_data import ingest_df

# chache is enabled coz if nothing changes in the data, the old data is used
@pipeline(enable_cache=True)
def train_pipeline(data_path:str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
