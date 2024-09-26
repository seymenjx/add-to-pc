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
import resource

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

def list_files_in_bucket_generator(bucket_name):
    """Generate a list of files in the specified S3 bucket."""
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket_name):
        for item in page.get('Contents', []):
            yield item['Key']

def read_file_from_s3_streaming(bucket, key):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = []
        for chunk in response['Body'].iter_chunks(chunk_size=4096):  # 4KB chunks
            if chunk:
                content.append(chunk.decode('utf-8'))
        if not content:
            logger.warning(f"File {key} is empty")
            return None
        return ''.join(content)
    except ClientError as e:
        logger.error(f"Error reading file {key}: {e}")
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

def semantic_documents_chunks_generator(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,    
        chunk_overlap=0,
        length_function=len,
    )
    for doc in documents:
        yield from text_splitter.split_documents([doc])

def add_documents_pinecone_batched(chunks, batch_size=100):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    pinecone_vs = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        pinecone_vs.add_documents(batch)
        logger.info(f"Added batch of {len(batch)} chunks to Pinecone index {INDEX_NAME}")

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

def limit_memory(max_mem):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_mem, hard))

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

def process_all_files(celery_task=None):
    try:
        limit_memory(450 * 1024 * 1024)
        
        file_generator = list_files_in_bucket_generator(os.getenv('AWS_BUCKET_NAME'))
        start_index = load_checkpoint()
        logger.info(f"Starting from index: {start_index}")

        total_files = sum(1 for _ in list_files_in_bucket_generator(os.getenv('AWS_BUCKET_NAME')))
        logger.info(f"Total files in bucket: {total_files}")

        processed_files = 0
        skipped_files = 0
        for i, key in enumerate(islice(file_generator, start_index, None), start=start_index):
            log_memory_usage()
            logger.info(f"Processing file {i+1} of {total_files}: {key}")
            
            if key in set(read_logs_from_s3()):
                logger.info(f'{key} already uploaded, skipping')
                skipped_files += 1
                continue
            
            main_content = read_file_from_s3_streaming(AWS_BUCKET_NAME, key)
            if main_content is None:
                logger.warning(f"Main content for {key} is empty or couldn't be read, skipping")
                skipped_files += 1
                continue

            summary_content = read_file_from_s3_streaming(AWS_BUCKET_NAME, f"summaries/{key}")
            if summary_content is None:
                logger.warning(f"Summary content for {key} is empty or couldn't be read, skipping")
                skipped_files += 1
                continue
            
            try:
                doc = docCreator(main_content, summary_content, key)
                chunks = list(semantic_documents_chunks_generator(doc))
                logger.info(f"Created {len(chunks)} chunks for {key}")
                
                for chunk in chunks:
                    add_documents_pinecone_batched([chunk])
                
                append_to_logs_s3(key)
                processed_files += 1
                
                progress = 100 * (i + 1) / total_files
                logger.info(f'{progress:.2f}% done. Processed {processed_files} files, skipped {skipped_files} files.')
                
                if celery_task:
                    celery_task.update_state(state='PROGRESS', meta={'status': f'{progress:.2f}% complete'})
            except Exception as e:
                logger.warning(f"Error processing {key}: {e}")
                skipped_files += 1
                continue
            
            gc.collect()
            time.sleep(1)
            
            if i % 10 == 0:
                save_checkpoint(i)
        
        logger.info(f"Processing complete. Processed {processed_files} files, skipped {skipped_files} files, out of {total_files} total files.")
        return {'status': 'All files processed successfully'}
    except Exception as e:
        logger.critical(f"Critical error in main process: {e}")
        return {'status': f'Error: {str(e)}'}
