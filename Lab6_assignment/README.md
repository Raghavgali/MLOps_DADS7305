## Cloud Run Intermediate Lab (FastAPI + IMDb Search)

This lab builds on the original *Cloud Run Intermediate Lab* but swaps Flask for **FastAPI** and showcases how to query the public **IMDb titles** dataset in BigQuery. The resulting container exposes a `/search` endpoint that lets you filter movies by title, release year, type, and genre. Everything else—containerization, deployment to Cloud Run, and Google Cloud IAM setup—follows the same flow as the base lab.

---

### Table of Contents
- [Prerequisites](#prerequisites)
- [Step 1: Configure Your Google Cloud Project](#step-1-configure-your-google-cloud-project)
- [Step 2: Build the FastAPI Application](#step-2-build-the-fastapi-application)
- [Step 3: Install Dependencies](#step-3-install-dependencies)
- [Step 4: Containerize With Docker](#step-4-containerize-with-docker)
- [Step 5: Push to Container Registry](#step-5-push-to-container-registry)
- [Step 6: Deploy on Cloud Run](#step-6-deploy-on-cloud-run)
- [Step 7: Test the FastAPI Endpoints](#step-7-test-the-fastapi-endpoints)
- [Step 8: Monitor and Clean Up](#step-8-monitor-and-clean-up)
- [Conclusion](#conclusion)

---

### Prerequisites
- Google Cloud project with billing enabled.
- `gcloud` CLI and Docker installed locally.
- Python 3.12+ for local development (FastAPI + uvicorn).
- Service account credentials with *BigQuery User* access when running locally.
- Basic familiarity with FastAPI (instead of Flask in the base lab).

---

### Step 1: Configure Your Google Cloud Project
1. **Create or select a project** in the [Cloud Console](https://console.cloud.google.com/).
2. **Enable APIs**: Cloud Run, Artifact/Container Registry, and BigQuery.
3. **Create a service account** (e.g., `cloud-run-fastapi-sa`) with:
   - `roles/bigquery.user`
   - (Optional) `roles/logging.logWriter` if you plan to write custom logs.
4. **Download a JSON key** if you want to run the app locally and export `GOOGLE_APPLICATION_CREDENTIALS=</path/to/key.json>`.

---

### Step 2: Build the FastAPI Application
The key difference from the base lab is the FastAPI app (`app.py`), which exposes:

- `GET /health` – lightweight health probe.
- `POST /search` – queries `bigquery-public-data.imdb.title_basics`.

```python
from fastapi import FastAPI
from google.cloud import bigquery
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Welcome to Movie Search API")

class SearchModel(BaseModel):
    movie_title: str
    release_year: Optional[int] = None
    title_type: Optional[str] = None
    genre: Optional[str] = None

@app.post("/search")
async def search_movies(payload: SearchModel):
    # Builds a parameterized query over the IMDb dataset
    ...
```

FastAPI handles validation, and the code dispatches the blocking BigQuery query inside a background thread so the async endpoint stays responsive.

---

### Step 3: Install Dependencies
`requirements.txt` covers FastAPI plus the Google Cloud SDKs:
```
fastapi
pydantic
dotenv
uvicorn
gunicorn
google-cloud-storage
google-cloud-bigquery
```

Install locally with:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Step 4: Containerize With Docker
The `Dockerfile` uses `python:3.12-slim`, installs dependencies, and starts uvicorn:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

Build the image (replace `YOUR_PROJECT_ID`):
```bash
docker build -t gcr.io/YOUR_PROJECT_ID/fastapi-imdb-search .
```

---

### Step 5: Push to Container Registry
```bash
gcloud auth configure-docker
docker push gcr.io/YOUR_PROJECT_ID/fastapi-imdb-search
```

---

### Step 6: Deploy on Cloud Run
```bash
gcloud run deploy fastapi-imdb-search \
  --image gcr.io/YOUR_PROJECT_ID/fastapi-imdb-search \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --service-account YOUR_SERVICE_ACCOUNT_EMAIL \
  --set-env-vars PORT=8080
```

Note: Unlike the original Flask app, there is no Cloud Storage upload endpoint—this build focuses entirely on querying the IMDb dataset with FastAPI’s request validation and async execution.

---

### Step 7: Test the FastAPI Endpoints
1. **Health check**
   ```bash
   curl https://YOUR_SERVICE_URL/health
   ```
2. **Movie search** (POST JSON body)
   ```bash
   curl -X POST https://YOUR_SERVICE_URL/search \
     -H "Content-Type: application/json" \
     -d '{
           "movie_title": "Matrix",
           "release_year": 1999,
           "title_type": "movie",
           "genre": "Action"
         }'
   ```
   Any field except `movie_title` can be omitted thanks to FastAPI/Pydantic defaults.

---

### Step 8: Monitor and Clean Up
- Use Cloud Run → **Logs** and **Metrics** tabs to observe requests.
- When finished, delete the service, image, and any temporary service accounts:
  ```bash
  gcloud run services delete fastapi-imdb-search --region us-central1
  gcloud container images delete gcr.io/YOUR_PROJECT_ID/fastapi-imdb-search --force-delete-tags
  gcloud iam service-accounts delete YOUR_SERVICE_ACCOUNT_EMAIL
  ```

---

### Roadblocks, Debugging Journey & How They Were Solved

1. **BigQuery Schema Mismatch (Column Not Found)**  
   - *Problem*: `Query parameter 'title_type' not found`, even though the column exists.  
   - *Cause*: The public IMDb schema uses different casing/naming than the tutorial.  
   - *Fix*: Inspected the dataset schema in the BigQuery console and aligned all field names (`primaryTitle`, `titleType`, etc.) within the SQL query.

2. **Container Failed to Start on Cloud Run**  
   - *Problem*: Logs showed `exec format error` and the container never bound to port 8080.  
   - *Cause*: Image built on Apple Silicon (ARM); Cloud Run expects `linux/amd64`.  
   - *Fix*: Rebuilt with `docker buildx build --platform linux/amd64 ...` before pushing.

3. **Missing Run Invoker Permissions**  
   - *Problem*: Visiting the service URL returned `Error: Forbidden`.  
   - *Cause*: Unauthenticated invoker role not granted.  
   - *Fix*: `gcloud run services add-iam-policy-binding ... --member=allUsers --role=roles/run.invoker`.

4. **Artifact Registry Permission Denied**  
   - *Problem*: `denied: Permission "artifactregistry.repositories.uploadArtifacts" denied`.  
   - *Cause*: Using a service account without Artifact Registry permissions.  
   - *Fix*: Granted Artifact Registry Writer, Storage Admin, and Cloud Run Admin; re-authenticated with `docker login` using an access token.

5. **Cloud Run Startup Probe Timeout**  
   - *Problem*: `Default STARTUP TCP probe failed`.  
   - *Cause*: Cold starts on the constrained ARM image caused slow boot.  
   - *Fix*: After rebuilding for amd64 (see #2) Cloud Run’s default probe succeeded; no custom probe needed.

6. **Local Docker Worked but Cloud Run Still Failed**  
   - *Problem*: Container responded locally yet still failed remotely.  
   - *Cause*: Same architecture mismatch as #2.  
   - *Fix*: Ensured every build pushed to Artifact Registry uses `--platform linux/amd64`.

7. **Service Account Could Not Deploy Cloud Run**  
   - *Problem*: `Permission 'iam.serviceaccounts.actAs' denied`.  
   - *Cause*: Missing Service Account User role on the deployment SA.  
   - *Fix*: `gcloud projects add-iam-policy-binding <PROJECT_ID> --member serviceAccount:<SA> --role roles/iam.serviceAccountUser`.

8. **BigQuery Credentials Failing Locally**  
   - *Problem*: `Default credentials not found`.  
   - *Cause*: No `GOOGLE_APPLICATION_CREDENTIALS` exported while testing outside Cloud Run.  
   - *Fix*: `export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"` before running `uvicorn`.

These notes capture the real-world troubleshooting steps for the FastAPI IMDb search service so future deployments go smoother.

---

### Conclusion
You now have the Cloud Run intermediate lab running on **FastAPI** with a dedicated `/search` endpoint that queries the public **IMDb** dataset via BigQuery. This variation highlights how to:
- Swap frameworks (FastAPI vs Flask) without changing the overall deployment workflow.
- Use Pydantic models to validate request bodies before querying.
- Keep async endpoints responsive by offloading BigQuery’s synchronous calls to a background thread.

Experiment with additional filters or pagination, or expand the API with Cloud Storage features similar to the base lab to explore more Google Cloud integrations.
