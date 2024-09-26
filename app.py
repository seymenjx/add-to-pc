from flask import Flask, request, jsonify
from tasks import start_processing

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process():
    task = start_processing.delay()
    return jsonify({"task_id": str(task.id)}), 202

if __name__ == '__main__':
    app.run(debug=True)
