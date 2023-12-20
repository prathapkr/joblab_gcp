from fastapi import FastAPI, HTTPException, UploadFile, File
import uuid
import requests
from werkzeug.utils import secure_filename
import json
import os
import tempfile

import config
import db
import openai

app = FastAPI()

database_file = "database.json"
database = db.load(database_file)
settings = config.load("settings.json")


# Function to find jobs
def custom_job_finder_state(query: str):
    url = "https://ai.joblab.ai/get_job_matches"
    query_params = {
        "query": query,
        "page": 1,
        "size": 5,
    }
    headers = {"accept": "application/json"}
    response = requests.post(url, params=query_params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        total_jobs = data["total"]
        job_matches_data = [
            {
                "job_id": job["job_id"],
                "job_title": job["job_title"],
                "job_company": job["job_company"],
                "job_location": job["job_location"],
                "job_description": job["job_description"],
            }
            for job in data["items"]
        ]
        return job_matches_data


@app.post("/chat", responses={200: {"model": ChatResponse}})
async def chat(chat: Chat):
    # Combine the functionality of new_chat, load_chat, send_message, and get_response here
    pass

@app.post("/upload", responses={200: {"model": UploadResponse}})
async def upload(file: UploadFile = File(...)):
    # Handle file uploads here
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
