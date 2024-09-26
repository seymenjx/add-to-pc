# Import necessary libraries
import boto3
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import json
import logging
import os
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import re
import gc
from itertools import islice
import psutil
import time
import platform
from memory_profiler import profile
from tenacity import retry, stop_after_attempt, wait_random_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# Validate required environment variables
required_env_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_BUCKET_NAME", "INDEX_NAME"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")

# Set AWS and Pinecone configuration variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
INDEX_NAME = os.getenv("INDEX_NAME")

# Initialize S3 client with error handling
try:
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {e}")
    raise

def list_files_in_bucket_generator(bucket_name, prefix='', batch_size=1000):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    
    kwargs = {'Bucket': bucket_name, 'Prefix': prefix}
    
    for page in paginator.paginate(**kwargs):
        batch = [item['Key'] for item in page.get('Contents', [])]
        if batch:
            yield batch

def process_file_batch(file_batch, embeddings):
    processed = 0
    skipped = 0
    for file_key in file_batch:
        try:
            if process_single_file(file_key, embeddings):
                processed += 1
            else:
                skipped += 1
        except Exception as e:
            logger.error(f"Error processing file {file_key}: {e}")
    return {'processed': processed, 'skipped': skipped}

def process_all_files(celery_task=None):
    try:
        total_files = get_total_file_count() or 0
        processed_files = 0
        skipped_files = 0
        start_index = load_checkpoint()
        logger.info(f"Starting from index: {start_index}")

        embeddings = create_embeddings()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for file_batch in list_files_in_bucket_generator(os.getenv('AWS_BUCKET_NAME')):
                futures.append(executor.submit(process_file_batch, file_batch, embeddings))
                
            for future in as_completed(futures):
                batch_result = future.result()
                processed_files += batch_result['processed']
                skipped_files += batch_result['skipped']
                
                if celery_task:
                    celery_task.update_state(state='PROGRESS', meta={
                        'current': processed_files + skipped_files,
                        'total': total_files,
                        'status': f'Processed: {processed_files}, Skipped: {skipped_files}'
                    })
                
                save_checkpoint(processed_files + skipped_files)
        
        logger.info(f"Processing complete. Processed {processed_files} files, skipped {skipped_files} files.")
        return {'status': 'All files processed successfully', 'processed': processed_files, 'skipped': skipped_files}
    except Exception as e:
        logger.critical(f"Critical error in main process: {str(e)}", exc_info=True)
        return {'status': f'Error: {str(e)}'}

def create_embeddings():
    return OpenAIEmbeddings(model='text-embedding-3-large')

def read_file_from_s3_streaming(bucket, key):
    try:
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket, Key=key)
        content = []
        for chunk in response['Body'].iter_chunks(chunk_size=4096):
            try:
                content.append(chunk.decode('utf-8'))
            except UnicodeDecodeError:
                # Try decoding with 'latin-1' if UTF-8 fails
                content.append(chunk.decode('latin-1'))
        return ''.join(content)
    except Exception as e:
        logger.error(f"Error reading file {key} from S3: {str(e)}")
        return None

def docCreator(main_content, summary_content, key):
    if not isinstance(main_content, str):
        logger.warning(f"main_content for {key} is not a string. Converting to string.")
        main_content = str(main_content) if main_content is not None else ""

    if summary_content:
        try:
            if not isinstance(summary_content, str):
                logger.warning(f"summary_content for {key} is not a string. Converting to string.")
                summary_content = str(summary_content)

            metadata = {
                "source": key,
                "summary": summary_content
            }
        except Exception as e:
            logger.error(f"Error processing summary content for {key}: {e}")
            metadata = {"source": key}
    else:
        metadata = {"source": key}
    
    logger.info(f"Created document with metadata for key: {key}")
    return Document(page_content=main_content, metadata=metadata)

def semantic_documents_chunks_generator(document):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,  # Reduced chunk size
        chunk_overlap=0,
        length_function=len,
    )
    for chunk in text_splitter.split_text(document.page_content):
        yield Document(page_content=chunk, metadata=document.metadata)

def process_chunk(chunk, embeddings):
    try:
        pinecone_vs = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
        pinecone_vs.add_documents([chunk])
        logger.info(f"Added chunk to Pinecone index {INDEX_NAME}")
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")

def process_single_file(key, embeddings):
    try:
        if key in set(read_logs_from_s3()):
            logger.info(f'{key} already uploaded, skipping')
            return False

        main_content = read_file_from_s3_streaming(AWS_BUCKET_NAME, key)
        if main_content is None:
            logger.warning(f"Main content for {key} is empty or couldn't be read, skipping")
            return False

        summary_content = read_file_from_s3_streaming(AWS_BUCKET_NAME, f"summaries/{key}")
        if summary_content is None:
            logger.warning(f"Summary content for {key} not found, proceeding with main content only")
            summary_content = ""

        doc = docCreator(main_content, summary_content, key)
        for chunk in semantic_documents_chunks_generator(doc):
            process_chunk(chunk, embeddings)
            gc.collect()

        append_to_logs_s3(key)
        return True
    except Exception as e:
        logger.warning(f"Error processing {key}: {str(e)}", exc_info=True)
        return False

def process_files_in_batches(file_list, batch_size=10):
    embeddings = create_embeddings()
    processed_files = 0
    skipped_files = 0

    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i+batch_size]
        for file in batch:
            log_memory_usage()
            logger.info(f"Processing file number {i+1}: {file}")
            
            if process_single_file(file, embeddings):
                processed_files += 1
            else:
                skipped_files += 1

            logger.info(f"Processed {processed_files} files, skipped {skipped_files} files.")

        # Force garbage collection after each batch
        gc.collect()

    return processed_files, skipped_files

def save_checkpoint(last_processed_index):
    """Save the current processing checkpoint to S3."""
    checkpoint_data = json.dumps({'last_processed_index': last_processed_index})
    s3.put_object(Bucket=AWS_BUCKET_NAME, Key='checkpoint.json', Body=checkpoint_data)

def load_checkpoint():
    try:
        response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key='checkpoint.json')
        checkpoint_data = json.loads(response['Body'].read().decode('utf-8'))
        return checkpoint_data.get('last_processed_index', 0)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.info("No checkpoint file found. Starting from the beginning.")
            return 0
        else:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error loading checkpoint: {str(e)}")
        raise

def read_logs_from_s3():
    """Read processing logs from S3."""
    try:
        response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key='logs.txt')
        return response['Body'].read().decode('utf-8').splitlines()
    except s3.exceptions.NoSuchKey:
        return []

def append_to_logs_s3(key):
    """Append a processed file key to the logs in S3."""
    current_logs = read_logs_from_s3()
    current_logs.append(key)
    logs_content = '\n'.join(current_logs)
    s3.put_object(Bucket=AWS_BUCKET_NAME, Key='logs.txt', Body=logs_content)

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def get_total_file_count(bucket_name):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    total_files = 0
    for page in paginator.paginate(Bucket=bucket_name):
        total_files += len(page.get('Contents', []))
    return total_files

def update_total_file_count(count):
    s3.put_object(Bucket=AWS_BUCKET_NAME, Key='total_file_count.txt', Body=str(count).encode('utf-8'))


