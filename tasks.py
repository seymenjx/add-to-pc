from celery import shared_task
import os
from dotenv import load_dotenv
from add_docs import process_all_files

# Load environment variables
load_dotenv(override=True)

@shared_task(bind=True)
def process_files(self):
    try:
        process_all_files(self)
        return {'status': 'All files processed successfully'}
    except Exception as e:
        return {'status': f'Error: {str(e)}'}
