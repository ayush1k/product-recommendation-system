# Product Recommendation System

This repository contains a small demo app: a React frontend (Vite) and a FastAPI backend that uses an LLM to provide product recommendations.

## Running locally

Prerequisites:
- Python 3.11+ and `venv`
- Node.js + npm
- A Google GenAI credential (recommended) or a Hugging Face token if using HF models.

Backend (FastAPI):

1. Create and activate a virtual environment:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables in `backend/.env` or your shell:

- For Google Gemini (recommended):
  - `GOOGLE_MODEL` (optional, default `gemini-2.5-flash`)
  - `GOOGLE_TEMPERATURE` (optional, default `0`)
  - Provide credentials via `GOOGLE_API_KEY` or `GOOGLE_APPLICATION_CREDENTIALS` pointing to a service-account JSON.

- (Alternative) For Hugging Face endpoint (not recommended here):
  - `HUGGINGFACEHUB_API_TOKEN`
  - `HUGGINGFACE_MODEL_REPO` to select an HF model

4. Start the backend:

```bash
# from backend/
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Frontend (React + Vite):

```bash
cd frontend
npm install
npm run dev
```

Open the dev URL printed by Vite (typically `http://localhost:5173`) and use the app.

## Notes
- The backend expects the LLM to extract structured constraints and return recommended product IDs. If the model is not accessible or fails, the API may return an empty list.
- Logs for constraint extraction errors are written to `backend/constraint_error.log` for debugging.