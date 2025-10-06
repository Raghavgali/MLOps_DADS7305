from __future__ import annotations

import os 
import time 
import pendulum
import requests
from airflow import DAG 
from airflow.providers.standard.operators.python import PythonOperator
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# ---------- Config (Airflow 3: use REST with Basic Auth via FAB API backend) ----------
WEBSERVER = os.getenv("AIRFLOW_WEBSERVER", "http://airflow-apiserver:8080")
AF_USER   = os.getenv("AIRFLOW_USERNAME", os.getenv("_AIRFLOW_WWW_USER_USERNAME", "airflow"))
AF_PASS   = os.getenv("AIRFLOW_PASSWORD", os.getenv("_AIRFLOW_WWW_USER_PASSWORD", "airflow"))
TARGET_DAG_ID = os.getenv("TARGET_DAG_ID", "Airflow_Lab2")


# ---------- Default args ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

# ------------ FastAPI app -----------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Utility function to get latest run info
def get_latest_run_info():
    """
    Query Airflow stable REST API (/api/v2) using Basic Auth.
    Requires Airflow to be configured with:
      AIRFLOW__FAB__AUTH_BACKENDS=airflow.providers.fab.auth_manager.api.auth.backend.basic_auth
    """
    url = f"{WEBSERVER}/api/v2/dags/{TARGET_DAG_ID}/dagRuns?order_by=-logical_date&limit=1"
    try:
        r = requests.get(url, auth=(AF_USER, AF_PASS), timeout=5)
    except Exception as e:
        return False, {"note": f"Exception calling Airflow API: {e}"}

    # If auth/backend is not set correctly you'll get 401 here.
    if r.status_code != 200:
        # Surface a short note (kept small to avoid template overflow)
        snippet = r.text[:200].replace("\n", " ")
        return False, {"note": f"API status {r.status_code}: {snippet}"}

    runs = r.json().get("dag_runs", [])
    if not runs:
        return False, {"note": "No DagRuns found yet."}

    run = runs[0]
    state = run.get("state")
    info = {
        "state": state,
        "run_id": run.get("dag_run_id"),
        "logical_date": run.get("logical_date"),
        "start_date": run.get("start_date"),
        "end_date": run.get("end_date"),
        "note": "",
    }
    return state == "success", info


# Home endpoint
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    run_info = get_latest_run_info()
    return templates.TemplateResponse("index.html", {"request": request, "run_info": run_info})

# Success endpoint
@app.get("/success", response_class=HTMLResponse)
async def success(request: Request):
    run_info = get_latest_run_info()
    return templates.TemplateResponse("success.html", {"request": request, "run_info": run_info})

# Failure endpoint
@app.get("/failure", response_class=HTMLResponse)
async def failure(request: Request):
    run_info = get_latest_run_info()
    return templates.TemplateResponse("failure.html", {"request": request, "run_info": run_info})

# Health endpoint
@app.get("/health", response_class=JSONResponse)
async def health():
    return {"status": "ok"}

# Function to start the FastAPI app (to be used in PythonOperator)
def start_fastapi_app():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ---------- DAG ----------
fast_api_dag = DAG(
    dag_id="Airflow_Lab2_Fastapi",
    default_args=default_args,
    description="DAG to manage FastAPI lifecycle",
    schedule=None,                 # trigger-only
    catchup=False,
    is_paused_upon_creation=False,
    tags=["Fast_Api"],
    max_active_runs=1,
)

start_fast_API = PythonOperator(
    task_id="start_fastapi_app",
    python_callable=start_fastapi_app,
    dag=fast_api_dag,
)

start_fastapi_app

if __name__ == "__main__":
    start_fast_API.cli()