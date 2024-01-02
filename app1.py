from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tempfile
import openai
import uuid
import os
import requests
import json

import config
import db


app = Flask(__name__)

database_file = "database.json"
database = db.load(database_file)
settings = config.load("settings.json")


# Custom function to find jobs
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


@app.route("/")
def index():
    return render_template("index.html")


# ... [Other routes like /new_chat, /load_chat, /conversations, /send_message]


@app.route("/new_chat", methods=["POST"])
def new_chat():
    chat_id = str(uuid.uuid4())

    thread = openai.beta.threads.create()

    chat = {
        "id": chat_id,
        "thread_id": thread.id,
        "title": "JobGPT chat",
    }

    database["conversations"][chat_id] = chat
    db.save(database_file, database)

    return render_template("chat_button.html", chat=chat)


@app.route("/load_chat/<chat_id>")
def load_chat(chat_id):
    thread_id = database["conversations"][chat_id]["thread_id"]

    messages = openai.beta.threads.messages.list(
        thread_id=thread_id,
        order="desc",
    )

    message_list = []

    for message in messages.data:
        message_list.append(
            {"role": message.role, "content": message.content[0].text.value}
        )

    message_list = reversed(message_list)

    return render_template("messages.html", messages=message_list, chat_id=chat_id)


@app.route("/conversations")
def conversations():
    chats = database["conversations"].values()
    return render_template("conversations.html", conversations=chats)


@app.route("/send_message", methods=["POST"])
def send_message():
    chat_id = request.form["chat_id"]
    file_ids = []

    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            temp_dir = tempfile.mkdtemp()

            filename = secure_filename(file.filename)
            file_path = os.path.join(temp_dir, filename)

            print(f"Saving to {file_path}")

            file.save(file_path)
            uploaded_file = openai.files.create(
                file=openai.file_from_path(file_path),
                purpose="assistants",
            )

            file_ids.append(uploaded_file.id)

            os.remove(file_path)
            os.rmdir(temp_dir)

    message = {"role": "user", "content": request.form["message"]}

    chat = database["conversations"][chat_id]

    # Add the message after handling the run
    openai.beta.threads.messages.create(
        thread_id=chat["thread_id"],
        role=message["role"],
        content=message["content"],
        file_ids=file_ids,
    )

    return render_template("user_message.html", chat_id=chat_id, message=message)


@app.route("/get_response/<chat_id>")
def get_response(chat_id):
    chat = database["conversations"][chat_id]

    # Create a new run
    run = openai.beta.threads.runs.create(
        thread_id=chat["thread_id"],
        assistant_id=settings["assistant_id"],
    )

    # Store the run_id in the chat object and save it
    chat["run_id"] = run.id
    db.save(database_file, database)

    # Retrieve the current run to check its status
    current_run = openai.beta.threads.runs.retrieve(
        run_id=run.id, thread_id=chat["thread_id"]
    )

    # Wait for the run to not be in an active state
    while current_run.status in ["queued", "in_progress", "cancelling"]:
        current_run = openai.beta.threads.runs.retrieve(
            run_id=run.id, thread_id=chat["thread_id"]
        )

        if current_run.status == "requires_action":
            tools_output = []
            for tool_call in current_run.required_action.submit_tool_outputs.tool_calls:
                f = tool_call.function
                if f.name == "custom_job_finder":
                    query = json.loads(f.arguments)["query"]
                    tool_result = custom_job_finder_state(query)
                    tool_result_json = json.dumps(tool_result)
                    tools_output.append(
                        {
                            "tool_call_id": tool_call.id,
                            "output": tool_result_json,
                        }
                    )

            openai.beta.threads.runs.submit_tool_outputs(
                thread_id=chat["thread_id"],
                run_id=run.id,
                tool_outputs=tools_output,
            )

            # Re-check the run status after handling requires_action
            current_run = openai.beta.threads.runs.retrieve(
                run_id=run.id, thread_id=chat["thread_id"]
            )

    # Retrieve the latest message after the run is completed or not active
    messages = openai.beta.threads.messages.list(
        thread_id=chat["thread_id"],
        order="desc",
        limit=1,
    )

    message = {"role": "assistant", "content": messages.data[0].content[0].text.value}
    return render_template("assistant_message.html", message=message)


if __name__ == "__main__":
    app.run(debug=True)
