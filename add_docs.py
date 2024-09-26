# Import necessary libraries
import boto3
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_ai21 import AI21SemanticTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
import json
import logging
import os
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from pinecone import PineconeException
import re
import gc
from itertools import islice
import psutil
import os

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

def list_files_in_bucket(bucket_name):
    """List all files in the specified S3 bucket."""
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name)
    return [item['Key'] for item in response.get('Contents', [])]

# Get list of files in the S3 bucket
files = list_files_in_bucket(os.getenv('AWS_BUCKET_NAME'))

def read_file_from_s3(bucket, key):
    """Read a file from S3 bucket."""
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        logger.info(f"Successfully read file: {key}")
        return content
    except ClientError as e:
        logger.error(f"Error reading file {key}: {e}")
        return None

def docCreator(main_content, summary_content, key):
    """Create a Document object with metadata."""
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

def clean_json(json_string):
    """Clean and format JSON string."""
    # Remove extra spaces and fix improperly escaped quotes
    json_string = re.sub(r'\s*"\s*', r'"', json_string)
    # Replace single quotes with double quotes
    json_string = re.sub(r"'", r'"', json_string)
    # Remove trailing commas
    json_string = re.sub(r",\s*([}\]])", r"\1", json_string)
    # Add missing commas between key-value pairs
    json_string = re.sub(r'"\s*:\s*"', r'": "', json_string)
    json_string = re.sub(r'"\s*:\s*([^"]+)\s*"', r'": "\1"', json_string)
    json_string = re.sub(r'"\s*:\s*([^"]+)\s*([}\]])', r'": "\1"\2', json_string)
    json_string = re.sub(r'([}\]])\s*([{"\[])', r'\1,\2', json_string)
    # Fix improperly escaped quotes
    json_string = re.sub(r'\\?"', r'"', json_string)
    return json_string

def semantic_documents_chunks(documents):
    """Split documents into semantic chunks."""
    logger.info('Starting document chunking...')
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,    
        chunk_overlap=0,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents=documents)
    logger.info(f"Created {len(chunks)} chunks from documents")
    return chunks

def add_documents_pinecone(chunks):
    """Add document chunks to Pinecone index."""
    try:
        embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        pinecone_vs = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
        pinecone_vs.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks to Pinecone index {INDEX_NAME}")
    except PineconeException as e:
        logger.error(f"Error adding documents to Pinecone: {e}")
        raise

def read_files_from_s3(bucket, key):
    """Read main content and summary files from S3."""
    try:
        main_content = s3.get_object(Bucket=bucket, Key=key)['Body'].read().decode('utf-8')
        summary_key = f"summaries/{key}"
        summary_content = s3.get_object(Bucket=bucket, Key=summary_key)['Body'].read().decode('utf-8')
        return main_content, summary_content
    except Exception as e:
        print(f"Error reading files: {e}")
        return None, None

def save_checkpoint(last_processed_index):
    """Save the current processing checkpoint to S3."""
    checkpoint_data = json.dumps({'last_processed_index': last_processed_index})
    s3.put_object(Bucket=AWS_BUCKET_NAME, Key='checkpoint.json', Body=checkpoint_data)

def load_checkpoint():
    """Load the last processing checkpoint from S3."""
    try:
        response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key='checkpoint.json')
        checkpoint_data = json.loads(response['Body'].read().decode('utf-8'))
        return checkpoint_data.get('last_processed_index', 0)
    except s3.exceptions.NoSuchKey:
        return 0

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

def list_files_in_bucket_generator(bucket_name):
    """Generate a list of files in the S3 bucket."""
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        for item in page.get('Contents', []):
            yield item['Key']

def read_file_from_s3_streaming(bucket, key):
    """Read a file from S3 in streaming mode."""
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        for chunk in response['Body'].iter_chunks(chunk_size=4096):  # 4KB chunks
            yield chunk.decode('utf-8')
    except ClientError as e:
        logger.error(f"Error reading file {key}: {e}")
        yield None

def semantic_documents_chunks_generator(documents):
    """Generate semantic chunks from documents."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,    
        chunk_overlap=0,
        length_function=len,
    )
    for doc in documents:
        yield from text_splitter.split_documents([doc])

def add_documents_pinecone_batched(chunks, batch_size=100):
    """Add document chunks to Pinecone index in batches."""
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    pinecone_vs = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        pinecone_vs.add_documents(batch)
        logger.info(f"Added batch of {len(batch)} chunks to Pinecone index {INDEX_NAME}")

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

def process_all_files(celery_task=None, batch_size=10):
    """Process all files in the S3 bucket and add them to Pinecone index."""
    try:
        file_generator = list_files_in_bucket_generator(os.getenv('AWS_BUCKET_NAME'))
        start_index = load_checkpoint()
        print(f"Starting from index: {start_index}")

        for i, batch in enumerate(iter(lambda: list(islice(file_generator, batch_size)), []), start=start_index):
            logs = set(read_logs_from_s3())
            for key in batch:
                if key in logs:
                    print(f'{key} already uploaded')
                    continue
                
                main_content = read_file_from_s3_streaming(AWS_BUCKET_NAME, key)
                summary_content = read_file_from_s3_streaming(AWS_BUCKET_NAME, f"summaries/{key}")
                
                if main_content is None or summary_content is None:
                    print(f"Skipping {key} due to read error or empty file")
                    continue
                
                try:
                    doc = docCreator(main_content, summary_content, key)
                    chunks = list(semantic_documents_chunks_generator([doc]))
                    add_documents_pinecone_batched(chunks)
                    append_to_logs_s3(key)
                    
                    progress = 100 * i * batch_size / len(list(list_files_in_bucket_generator(os.getenv('AWS_BUCKET_NAME'))))
                    print(f'{progress:.2f}% done')
                    
                    if celery_task:
                        celery_task.update_state(state='PROGRESS', meta={'status': f'{progress:.2f}% complete'})
                except Exception as e:
                    logger.warning(f"Error processing {key}: {e}")
                    continue
                
                gc.collect()  # Force garbage collection after processing each file
            
            save_checkpoint(i * batch_size)
        
        print("Processing complete")
        return {'status': 'All files processed successfully'}
    except Exception as e:
        logger.critical(f"Critical error in main process: {e}")
        return {'status': f'Error: {str(e)}'}

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # in MB
    logger.info(f"Current memory usage: {mem:.2f} MB")

# Call this function periodically in your process_all_files function
