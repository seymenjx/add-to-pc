
import boto3
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_ai21 import AI21SemanticTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document


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
    page_iterator = paginator.paginate(Bucket=bucket_name)
    
    files = []
    for page in page_iterator:
        files.extend([obj['Key'] for obj in page.get('Contents', [])])
    
    return files

def download_file(bucket_name, key, local_dir):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    local_file_path = os.path.join(local_dir, os.path.basename(key))
    s3.download_file(bucket_name, key, local_file_path)
    print(f"Downloaded {key} to {local_file_path}")
    return local_file_path

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

    ''' for chunk in chunks:
        f = chunk.metadata['source']
        with open(f, 'r', encoding='utf-8') as file:
                    count = 0
                    for l in file.readlines():
                        count += 1
                        if l.startswith('Esas :'):
                            chunk.metadata['esas'] = l.replace("\n", "")
                        elif l.startswith('Karar :'):
                            chunk.metadata['karar'] = l.replace("\n", "")
    '''
    return chunks

def add_documents_pinecone(chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    pinecone_vs = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    pinecone_vs.add_documents(chunks)


files= list_files(bucket_name=AWS_BUCKET_NAME, prefix=PREFIX)
print(len(files))
'''
for i, key in enumerate(files):
    with open('logs.txt', 'r', encoding='utf-8') as controller:
        lines = controller.readlines()
    if str(key+'\n') in lines:
        print('already uploaded')
        continue
    else:
        file_path = download_file(bucket_name=AWS_BUCKET_NAME, key=key, local_dir='downloaded')

        doc = docCreator(file_path)
        chunks = semantic_documents_chunks(doc)
        add_documents_pinecone(chunks=chunks)

        with open('logs.txt', '+a', encoding='utf-8') as f:
            f.write(key)
            f.write('\n')
        print(f'%{100*i/len(files)} done')
        os.remove(file_path)
'''

for i, key in enumerate(files):
    with open('logs.txt', 'r', encoding='utf-8') as controller:
        lines = controller.readlines()
    if str(key+'\n') in lines:
        print('already uploaded')
        continue
    else:
        content= read_file_from_s3(bucket=AWS_BUCKET_NAME, key=key)
        try: 
            doc =  [Document(page_content=content, metadata={"source": key})]
            chunks = semantic_documents_chunks(doc)
            add_documents_pinecone(chunks=chunks)
            with open('logs.txt', '+a', encoding='utf-8') as f:
                f.write(key)
                f.write('\n')
                print(f'%{100*i/len(files)} done')
        except Exception as e:
            print(f'error:{e}')
            continue