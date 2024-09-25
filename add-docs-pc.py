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
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name)
    return [item['Key'] for item in response.get('Contents', [])]

files = list_files_in_bucket(os.getenv('AWS_BUCKET_NAME'))

def read_file_from_s3(bucket, key):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        logger.info(f"Successfully read file: {key}")
        return content
    except ClientError as e:
        logger.error(f"Error reading file {key}: {e}")
        return None

def docCreator(main_content, summary_content, key):
    if summary_content:
        try:
                 # If summary_content is a string, convert it to a dictionary
            """if isinstance(summary_content, str):
                cleaned_summary_content = clean_json(summary_content)
                print(cleaned_summary_content)
                summary_dict = json.loads(cleaned_summary_content)
                
            else:
                summary_dict = summary_content"""

            metadata = {
                "source": key,
                "summary": summary_content
            }
            """
                "dava_konusu": summary_dict.get("Dava Konusu", ""),
                "hukuki_dayanak": summary_dict.get("Hukuki Dayanak", ""),
                "mahkeme_karari": summary_dict.get("Mahkeme Kararı", ""),
                "kararin_gerekcesi": summary_dict.get("Kararın Gerekçesi", ""),
                "output": summary_dict.get(" Output", "")
                """
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing summary content for {key}: {e}")
            metadata = {"source": key}
    else:
        metadata = {"source": key}
    
    logger.info(f"Created document with metadata for key: {key}")
    return Document(page_content=main_content, metadata=metadata)

def clean_json(json_string):
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
    try:
        embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        pinecone_vs = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
        pinecone_vs.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks to Pinecone index {INDEX_NAME}")
    except PineconeException as e:
        logger.error(f"Error adding documents to Pinecone: {e}")
        raise

def read_files_from_s3(bucket, key):
    try:
        main_content = s3.get_object(Bucket=bucket, Key=key)['Body'].read().decode('utf-8')
        summary_key = f"summaries/{key}"
        summary_content = s3.get_object(Bucket=bucket, Key=summary_key)['Body'].read().decode('utf-8')
        return main_content, summary_content
    except Exception as e:
        print(f"Error reading files: {e}")
        return None, None

def save_checkpoint(last_processed_index):
    checkpoint_data = json.dumps({'last_processed_index': last_processed_index})
    s3.put_object(Bucket=AWS_BUCKET_NAME, Key='checkpoint.json', Body=checkpoint_data)

def load_checkpoint():
    try:
        response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key='checkpoint.json')
        checkpoint_data = json.loads(response['Body'].read().decode('utf-8'))
        return checkpoint_data.get('last_processed_index', 0)
    except s3.exceptions.NoSuchKey:
        return 0

def read_logs_from_s3():
    try:
        response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key='logs.txt')
        return response['Body'].read().decode('utf-8').splitlines()
    except s3.exceptions.NoSuchKey:
        return []

def append_to_logs_s3(key):
    current_logs = read_logs_from_s3()
    current_logs.append(key)
    logs_content = '\n'.join(current_logs)
    s3.put_object(Bucket=AWS_BUCKET_NAME, Key='logs.txt', Body=logs_content)

if __name__ == '__main__':
    try:
        files = list_files_in_bucket(os.getenv('AWS_BUCKET_NAME'))
        logger.info(f"Total files to process: {len(files)}")

        # Load the last processed index from the checkpoint
        start_index = load_checkpoint()
        print(f"Starting from index: {start_index}")

        for i, key in enumerate(files[start_index:], start=start_index):
            logs = read_logs_from_s3()
            if key in logs:
                print(f'{key} already uploaded')
                continue
            else:
                main_content, summary_content = read_files_from_s3(bucket=AWS_BUCKET_NAME, key=key)
                if main_content is None or summary_content is None:
                    print(f"Skipping {key} due to read error")
                    continue
                try: 
                    doc = [docCreator(main_content, summary_content, key)]
                    chunks = semantic_documents_chunks(doc)
                    add_documents_pinecone(chunks=chunks)
                    append_to_logs_s3(key)
                    print(f'%{100*i/len(files)} done')
                    
                    # Save checkpoint every 100 documents
                    if i % 100 == 0:
                        save_checkpoint(i)
                except Exception as e:
                    print(f'Error processing {key}: {e}')
                    continue

        # Save final checkpoint
        save_checkpoint(len(files))
        print("Processing complete")
    except Exception as e:
        logger.critical(f"Critical error in main process: {e}")
        raise
