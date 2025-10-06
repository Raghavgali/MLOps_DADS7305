from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BaseOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.task.trigger_rule import TriggerRule


from src.model_development import (
    load_data,
    data_preprocessing,
    separate_data_outputs,
    build_model,
    load_model,
)

# ---------- Default args ----------
default_args = {
    "start_date": pendulum.datetime(2025, 1, 1, tz="UTC"),
    "retries": 0
}

# ---------- DAG ----------
dag = DAG(
    dag_id="Airflow_Lab2",
    default_args=default_args,
    description="Airflow-Lab2 DAG Description",
    schedule="@daily",
    catchup=False,
    tags=["example"],
    owner_links={"Raghavgali": ""}
)

