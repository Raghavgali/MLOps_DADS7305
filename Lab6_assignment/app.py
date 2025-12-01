from fastapi import FastAPI
from google.cloud import bigquery
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import uvicorn
import os
import asyncio

load_dotenv()

app = FastAPI(title="Welcome to Movie Search API")

class search_model(BaseModel):
    movie_title: str


@app.get("/health")
async def get_health():
    return {"status": "Health Check ok"}


@app.post("/search")
async def search_movies(search_params: search_model):
    movie_title = search_params.movie_title

    query = """
        SELECT
            primary_title,
            start_year,
            title_type,
            genres
        FROM `bigquery-public-data.imdb.title_basics`
        WHERE LOWER(primary_title) LIKE LOWER(@movie_title)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("movie_title", "STRING", f"%{movie_title}%")
        ]
    )

    def execute_query():
        client = bigquery.Client()
        query_job = client.query(query, job_config=job_config)
        return list(query_job.result())

    results = await asyncio.to_thread(execute_query)

    movies = []
    for row in results:
        movies.append(
            {
                "title": row.primary_title,
                "year": row.start_year,
                "type": row.title_type,
                "genres": row.genres,
            }
        )

    return {"results": movies}
