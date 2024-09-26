from flask import Flask, request, jsonify
from tasks import start_processing
from celery.result import AsyncResult

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process():
    try:
        task = start_processing.delay()
        return jsonify({"task_id": str(task.id)}), 202
    except Exception as e:
        app.logger.error(f"Error starting processing task: {str(e)}")
        return jsonify({"error": "Failed to start processing task"}), 500

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task = AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Task is waiting for execution'
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
