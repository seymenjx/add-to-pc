from celery import Celery
import os
from add_docs import process_all_files
import logging

app = Celery('tasks')
app.config_from_object('celeryconfig')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.task(bind=True)
def start_processing(self, prefix=''):
    try:
        result = process_all_files(celery_task=self, prefix=prefix)  # Pass prefix
        return result
    except Exception as e:
        logger.error(f"Error in start_processing task: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise
