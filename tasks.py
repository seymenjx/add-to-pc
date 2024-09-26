from celery import Celery
import os

app = Celery('tasks')
app.config_from_object('celeryconfig')

# Import your tasks here
from add_docs import process_all_files

@app.task
def start_processing():
    return process_all_files()
