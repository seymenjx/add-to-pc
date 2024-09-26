from flask import Flask, jsonify, request
from celery import Celery
import os
from dotenv import load_dotenv
from tasks import process_files

# Load environment variables
load_dotenv(override=True)

app = Flask(__name__)

# Configure Celery
app.config['CELERY_BROKER_URL'] = os.getenv('REDIS_URL')
app.config['CELERY_RESULT_BACKEND'] = os.getenv('REDIS_URL')

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Move this import to the top of the file


@app.route('/process', methods=['POST'])
def start_processing():
    task = process_files.delay()
    return jsonify({"task_id": task.id, "status": "Processing started"}), 202

@app.route('/status/<task_id>')
def get_status(task_id):
    task = process_files.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'status': task.info.get('status', '')
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)
        }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
