# Breast-Cancer-MLOps
Using ZenML to build a pipeline

## Usage:
```bash
pip install -r requirements.txt
```
After that 
install the mlflow 

```bash
zenml integration install mlflow
```

```bash
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

Running Pipeline
```bash
python run_pipeline.py
```


