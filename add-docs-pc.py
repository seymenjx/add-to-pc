import boto3
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_ai21 import AI21SemanticTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv

load_dotenv(override=True)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
PREFIX =  os.getenv("PREFIX")
INDEX_NAME = os.getenv("INDEX_NAME")

s3 = boto3.client('s3', aws_access_key_id= AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

def list_files(bucket_name, prefix):
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    files = []
    for page in page_iterator:
        files.extend([obj['Key'] for obj in page.get('Contents', [])])
    
    return files

def read_file_from_s3(bucket, key):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        print(f"Dosya başarıyla okundu: {key}")
        return content
    except Exception as e:
        print(f"Dosya okunurken hata oluştu: {e}")
        return None

def docCreator(path):
    loader = TextLoader(path)
    return loader.load()

def semantic_documents_chunks(documents):
    print('chunking...')
    text_splitter = CharacterTextSplitter(
            separator= "\n",
            chunk_size= 15000,
            chunk_overlap=0,
            length_function= len,
        )
    chunks1 = text_splitter.split_documents(documents=documents)
    semantic_text_splitter = AI21SemanticTextSplitter()
    chunks = semantic_text_splitter.split_documents(chunks1)
    return chunks

def add_documents_pinecone(chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    pinecone_vs = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    pinecone_vs.add_documents(chunks)

def process_file(key):
    with open('logs.txt', 'r', encoding='utf-8') as controller:
        lines = controller.readlines()
    if str(key+'\n') in lines:
        print('already uploaded')
        return
    
    content = read_file_from_s3(bucket=AWS_BUCKET_NAME, key=key)
    if content:
        try:
            doc = [Document(page_content=content, metadata={"source": key})]
            chunks = semantic_documents_chunks(doc)
            add_documents_pinecone(chunks=chunks)
            with open('logs.txt', '+a', encoding='utf-8') as f:
                f.write(key)
                f.write('\n')
            print(f'{key} uploaded')
        except Exception as e:
            print(f'error processing {key}: {e}')

def process_files_in_batches(files, batch_size=10):
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_key = {executor.submit(process_file, key): key for key in batch}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    future.result()
                except Exception as e:
                    print(f'error processing {key}: {e}')

def main():
    files = list_files(bucket_name=AWS_BUCKET_NAME, prefix=PREFIX)
    print(f'Total files: {len(files)}')
    process_files_in_batches(files)

if __name__ == "__main__":
    main()
