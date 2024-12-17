# Tracking server with MLFlow
[[_TOC_]]

Setup to tracking multiple ML experiments with `mlflow`

## Server

### Add to `${HOME}/.bashrc`

```shell
# --- GCS access
export GOOGLE_APPLICATION_CREDENTIALS="path/to/admin.json"

# --- MLFlow tracking
export MFLOW_LOGGING_DB_PATH="${HOME}/mlflow_logging.db"
export MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT=300
```

### Setup

First

```shell
cd $HOME
python3.10 -m venv python-venv
source python-venv/bin/activate
pip install -U pip
pip install poetry==1.8.5
```

Then

```shell
cd mlops/mlflow
poetry install
```

### Run

```shell
gsutil ls gs://llm-mlflow-artifacts
nohup mlflow server \
    --host 0.0.0.0 \
    --port 5051 \
    --backend-store-uri sqlite:///${MFLOW_LOGGING_DB_PATH} \
    --artifacts-destination gs://llm-mlflow-artifacts \
    > ${HOME}/mlflow_tracking_server.log 2>&1 &
```

### UI

On the server

```shell
mlflow ui --backend-store-uri sqlite:///${MFLOW_LOGGING_DB_PATH}
```

## Client

### Add to `${HOME}/.bashrc`

```shell
export MLFLOW_TRACKING_URI=http://cluster-manager:5051
```

## UI
On your laptop run

```shell
gcloud compute ssh \
    n2-cpuonly-llm-01-gabriele-cluster-manager-cont \
    --zone=us-central1-f \
    -- -NL 5000:localhost:5000
```

to serve the `mlflow` ui on http://localhost:5000