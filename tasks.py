from celery import Celery
import os
from dotenv import load_dotenv
from add_docs import process_all_files

# Load environment variables
load_dotenv(override=True)

# Initialize Celery
celery = Celery('tasks', broker=os.getenv('REDIS_URL'))

@celery.task(bind=True)
def process_files(self):
    try:
        process_all_files(self)
        return {'status': 'All files processed successfully'}
    except Exception as e:
        return {'status': f'Error: {str(e)}'}
