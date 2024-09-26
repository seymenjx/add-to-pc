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
from memory_profiler import profile
from tenacity import retry, stop_after_attempt, wait_random_exponential

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

def add_documents_pinecone_batched(chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    pinecone_vs = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    pinecone_vs.add_documents(chunks)
    logger.info(f"Added {len(chunks)} chunks to Pinecone index {INDEX_NAME}")

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

def get_total_file_count():
    try:
        response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key='total_file_count.txt')
        return int(response['Body'].read().decode('utf-8'))
    except:
        return None

def update_total_file_count(count):
    s3.put_object(Bucket=AWS_BUCKET_NAME, Key='total_file_count.txt', Body=str(count).encode('utf-8'))

def process_all_files(celery_task=None):
    try:
        limit_memory(450 * 1024 * 1024)
        
        file_generator = list_files_in_bucket_generator(os.getenv('AWS_BUCKET_NAME'))
        start_index = load_checkpoint()
        logger.info(f"Starting from index: {start_index}")

        processed_files = 0
        skipped_files = 0

        embeddings = create_embeddings()

        for i, key in enumerate(file_generator):
            if i < start_index:
                continue

            log_memory_usage()
            logger.info(f"Processing file number {i+1}: {key}")

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
                logger.warning(f"Summary content for {key} not found, proceeding with main content only")
                summary_content = ""
            
            try:
                doc = docCreator(main_content, summary_content, key)
                for chunk in semantic_documents_chunks_generator(doc):
                    process_chunk(chunk, embeddings)
                    gc.collect()

                append_to_logs_s3(key)
                processed_files += 1
                
                logger.info(f"Processed {processed_files} files, skipped {skipped_files} files.")
                
                if celery_task:
                    celery_task.update_state(state='PROGRESS', meta={'processed': processed_files, 'skipped': skipped_files})
            except Exception as e:
                logger.warning(f"Error processing {key}: {str(e)}", exc_info=True)
                skipped_files += 1
                continue
            
            gc.collect()
            time.sleep(1)
            
            if i % 10 == 0:
                save_checkpoint(i)
        
        logger.info(f"Processing complete. Processed {processed_files} files, skipped {skipped_files} files.")
        return {'status': 'All files processed successfully', 'processed': processed_files, 'skipped': skipped_files}
    except Exception as e:
        logger.critical(f"Critical error in main process: {str(e)}", exc_info=True)
        return {'status': f'Error: {str(e)}'}

@profile
def process_files_in_batches(file_list, batch_size=10):
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i+batch_size]
        try:
            process_batch(batch)
        except MemoryError:
            print(f"Memory error processing batch {i//batch_size + 1}. Skipping.")
        finally:
            # Force garbage collection
            gc.collect()

def process_batch(batch):
    for file in batch:
        try:
            # Your existing processing logic here
            process_single_file(file)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def create_embeddings():
    return OpenAIEmbeddings(model='text-embedding-3-large')

def process_chunk(chunk, embeddings):
    try:
        pinecone_vs = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
        pinecone_vs.add_documents([chunk])
        logger.info(f"Added chunk to Pinecone index {INDEX_NAME}")
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")

if __name__ == "__main__":
    all_files = get_all_files()  # Your existing method to get files
    process_files_in_batches(all_files)
