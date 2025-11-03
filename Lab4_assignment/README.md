# Lab 4 – CI/CD With GitHub Actions And Google Cloud

This lab walks through building a lightweight MLOps pipeline: train a PyTorch model, version it in Google Cloud Storage (GCS), package the project inside Docker, and ship the image to Artifact Registry from a GitHub Actions workflow. The repo mirrors the end state you are expected to reach while completing the lab.

---
## Learning Objectives
- Provision GCP services (GCS + Artifact Registry) that back an ML release pipeline.
- Configure GitHub Actions to authenticate against GCP and automate testing, training, image builds, and pushes.
- Version trained models in GCS and surface the version for subsequent deployment steps.
- Exercise unit tests that rely on `unittest.mock` to isolate GCP calls.

---
## Repository Layout
- `src/train_and_save_model.py` – end-to-end pipeline that trains a CNN on the scikit-learn Digits dataset, evaluates it, stores the artifact locally, uploads it to GCS, and bumps a version file.
- `test/` – pytest suite covering data prep, model behaviour, and the GCS helper utilities with MagicMock.
- `Dockerfile` / `.dockerignore` – image definition for production runs; keeps virtualenvs, caches, and secrets out of the context.
- `.github/workflows/ci_cd_pipeline.yaml` – GitHub Actions workflow that runs on pushes, PRs to `main`, nightly cron, and manual dispatch.

---
## Prerequisites On Google Cloud
1. Create or select a GCP project.
2. Enable **Cloud Storage**, **Artifact Registry**, and **Cloud Build** APIs.
3. Create a service account with roles:
   - Storage Admin
   - Storage Object Admin
   - Artifact Registry Administrator
4. Generate a JSON key for that service account.
5. Create a regional Artifact Registry repository for Docker images (for example `us-east4`).
6. Create a GCS bucket that will hold the trained models and the version file.

Add these GitHub repository secrets (Settings → Secrets and variables → Actions):
- `GCP_SA_KEY` – contents of the service-account JSON key.
- `GCP_PROJECT_ID` – your project ID.
- `GCP_BUCKET_NAME` – bucket used for artifacts.
- `VERSION_FILE_NAME` – filename that stores the current model version (example `model_version.txt`).

---
## Local Development
```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` in `Lab4_assignment/`:
```
GCS_BUCKET_NAME=your-bucket-name
VERSION_FILE_NAME=model_version.txt
```

Authenticate locally so Google client libraries can reach your project:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/service-account.json"
```

Run the training script:
```bash
python src/train_and_save_model.py
```
On each run the script increments the version file in GCS, writes a timestamped `trained_models/model_v<version>_<timestamp>.joblib`, and prints the new version as `MODEL_VERSION_OUTPUT: <n>`.

> **Tip:** GitHub-hosted runners have limited disk space. If you do not need CUDA, pin a CPU wheel such as `torch==2.3.1` from the official CPU index to keep installs light:
> ```bash
> pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
> ```

---
## Training Script Overview (`src/train_and_save_model.py`)
- `download_data()` – loads the handwritten digits dataset from scikit-learn.
- `preprocess_data()` – stratified split, standardisation, and reshape to `(batch, 1, 8, 8)` tensors.
- `data_loader()` – wraps NumPy arrays inside `TensorDataset` and `DataLoader`.
- `TrainMNIST` – simple convolutional network producing logits for 10 classes.
- `train_model()` / `evaluate_model()` – train for `num_epochs`, then compute accuracy on the test loader.
- `get_model_version()` / `update_model_version()` – fetch and store the integer version in GCS.
- `ensure_folder_exists()` / `save_model_to_gcs()` – guarantee a `trained_models/` prefix and upload the `.joblib` artifact.
- `main()` – orchestrates the steps, builds the destination blob name with a timestamp, persists the model, updates the version, and emits the new version to stdout.

---
## Docker Image
The image is based on `python:3.10-slim`, installs `requirements.txt`, copies the repo into `/app`, and runs `python src/train_and_save_model.py` as the container entrypoint. Use `.dockerignore` to keep local virtual environments, caches, and secrets out of the build context.

Example build (authenticated with GCP first):
```bash
docker build -t us-east4-docker.pkg.dev/<project>/<repo>/model-image:local .
```

---
## GitHub Actions Workflow Highlights
Trigger conditions: push to `main`, pull requests targeting `main`, nightly cron, or manual dispatch.

Key steps:
1. **Checkout** source code.
2. **Set up Python** 3.10 and restore pip cache.
3. **Install dependencies** and run `pytest`.
4. **Authenticate with GCP** using `google-github-actions/auth@v2` (reads `GCP_SA_KEY`).
5. **Configure gcloud** and Docker to target your Artifact Registry region.
6. **Train and save model** – runs the pipeline, writes stdout to `output.txt`, and exports `MODEL_VERSION` by parsing the printed `MODEL_VERSION_OUTPUT`.
7. **Build & push Docker image** – tags the image with the model version and `latest`, then pushes both tags.

Successful runs publish:
- A new model artifact and updated version file in your GCS bucket.
- A Docker image in Artifact Registry tagged with both the semantic version (`model-image:<n>`) and `model-image:latest`.

---
## Triggering The Pipeline
Commit changes inside `Lab4_assignment/` and push to `main`:
```bash
git add .
git commit -m "Finish Lab 4 automation"
git push origin main
```

To retrigger without code changes:
```bash
git commit --allow-empty -m "Kick CI for Lab 4"
git push
```

Monitor the Actions tab for status. After a green build, verify:
- `trained_models/` and the version file updated in the configured GCS bucket.
- `model-image` repository in Artifact Registry contains the new version and `latest` tags.

---
## Troubleshooting
- **Authentication step fails:** confirm the workflow is running on a branch that has access to repository secrets and that the service-account JSON is valid.
- **Bucket name errors:** ensure `GCP_BUCKET_NAME` secret matches the bucket exactly; the workflow exports it as `GCS_BUCKET_NAME` for the script.
- **Disk space errors during Docker build:** prefer CPU-only PyTorch wheels or build on a self-hosted runner with more storage.

With these pieces in place you have a reproducible CI/CD pipeline that trains, versions, packages, and publishes an ML model end-to-end using GitHub Actions and GCP.
