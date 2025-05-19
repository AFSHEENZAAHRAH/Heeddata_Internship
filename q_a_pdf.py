import fitz
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, BartTokenizer, BartForConditionalGeneration
import torch
import numpy as np
import uuid
from pymongo import MongoClient
from urllib.parse import quote_plus
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import re

name = 'afsheenzaahrah25'
pass_word = '9io1i0I1SlZvIZwZ'
username = quote_plus(name)
password = quote_plus(pass_word)
cluster = 'cluster0.jwxhgrj.mongodb.net'
database_name = 'third_database'
MONGO_URL = f'mongodb+srv://{username}:{password}@{cluster}/{database_name}?retryWrites=true'

try:
    mongo_client = MongoClient(MONGO_URL)
    db = mongo_client[database_name]
    collection_name = "third_data"
    collection = db[collection_name]
    print("Connected to MongoDB.")
except Exception as e:
    print(f'Error connecting to MongoDB: {str(e)}')

embedding_model_name = "all-mpnet-base-v2"
embedding_model = SentenceTransformer(embedding_model_name)

qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

summarization_model_name = "facebook/bart-large-cnn"
summarization_tokenizer = BartTokenizer.from_pretrained(summarization_model_name)
summarization_model = BartForConditionalGeneration.from_pretrained(summarization_model_name)

API_KEY = 'AIzaSyClSMtzoaYW0DuMZ13RasGnadzJbTB60wM'
CX = '17018756cc3ef430b'

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        document = fitz.open(pdf_path)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text() + "\n"
        text = preprocess_text(text)
        print(f"Extracted text (first 1000 chars): {text[:1000]}")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None

def get_embeddings(text, model):
    try:
        embeddings = model.encode(text, convert_to_tensor=True)
        print(f"Text: {text[:100]}... Embeddings shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return None

def split_text_into_chunks(text, max_chunk_size=512):
    try:
        sentences = text.split('. ')
        chunks = []
        chunk = ""
        for sentence in sentences:
            if len(chunk) + len(sentence) > max_chunk_size:
                chunks.append(chunk.strip())
                chunk = sentence + '. '
            else:
                chunk += sentence + '. '
        if chunk:
            chunks.append(chunk.strip())
        print(f"Total chunks created: {len(chunks)}")
        return chunks
    except Exception as e:
        print(f"Error splitting text into chunks: {str(e)}")
        return []

def summarize_text(text, max_length=150):
    try:
        inputs = summarization_tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = summarization_model.generate(inputs['input_ids'], max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4)
        summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(f"Summarized text: {summary[:500]}...")
        return summary
    except Exception as e:
        print(f"Error summarizing text: {str(e)}")
        return text

def save_to_mongodb(text_chunks, model):
    for i, text_chunk in enumerate(text_chunks):
        unique_id = str(uuid.uuid4())
        embeddings = get_embeddings(text_chunk, model)
        if embeddings is None:
            print(f"Skipping chunk {i} due to embedding error.")
            continue
        embeddings_list = embeddings.numpy().tolist()
        print(f"Inserting document {unique_id} with embeddings sample {embeddings_list[:5]}...")
        document = {
            'id': unique_id,
            'source': 'PDF',
            'data': text_chunk,
            'embeddings': embeddings_list
        }
        try:
            collection.insert_one(document)
            print(f"Document {unique_id} inserted.")
        except Exception as e:
            print(f"Error inserting document {unique_id}: {str(e)}")
    print("All documents inserted into MongoDB.")

def get_full_answer(context, question):
    try:
        result = qa_pipeline(question=question, context=context, max_length=512, stride=256)
        print(f"QA Pipeline Result: {result}")
        return result['answer']
    except Exception as e:
        print(f"Error getting answer from QA model: {str(e)}")
        return "No answer found."

def find_most_similar_chunks(question, chunks, model):
    question_embedding = get_embeddings(question, model)
    if question_embedding is None:
        print("Error generating embedding for the question.")
        return None

    chunk_embeddings = [get_embeddings(chunk, model) for chunk in chunks]
    similarities = [
        cosine_similarity(question_embedding.numpy().reshape(1, -1), chunk_embedding.numpy().reshape(1, -1))[0][0]
        if chunk_embedding is not None else 0
        for chunk_embedding in chunk_embeddings
    ]
    best_chunk_indices = np.argsort(similarities)[-3:] 
    best_chunk_indices = sorted(best_chunk_indices)

    combined_chunks = ' '.join(chunks[i] for i in best_chunk_indices)
    combined_chunks = summarize_text(combined_chunks)  
    print(f"Best chunk indices: {best_chunk_indices}, Combined summarized context: {combined_chunks[:500]}...")

    return combined_chunks

def get_embedding(text):
    try:
        embedding = get_embeddings(text, embedding_model)
        return embedding.numpy().tolist()
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def vector_search(user_query, collection):
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        print("Invalid query or embedding generation failed.")
        return []

    flattened_embedding = [float(item) for item in query_embedding]

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embeddings",
                "queryVector": flattened_embedding,
                "numCandidates": 100,
                "limit": 100
            }
        }
    ]

    try:
        results = collection.aggregate(pipeline)
        results_list = list(results)
        print(f"Vector search results: {results_list}")
        return results_list
    except Exception as e:
        print(f"Error during vector search: {str(e)}")
        return []

def search_google(query):
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': API_KEY,
        'cx': CX,
        'q': query,
        'num': 1  
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        search_results = response.json()

        print("Google API response:", search_results)
        if 'items' in search_results:
            return search_results['items'][0]['snippet']
        else:
            return "No information available on this topic."
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Google API: {str(e)}")
        return None

def handle_user_query(query, collection):
    print(f"Handling user query: {query}")

    # Search in the MongoDB collection (PDF data) first
    pdf_results = vector_search(query, collection)
    if pdf_results:
        chunks = [res['data'] for res in pdf_results]
        most_relevant_chunk = find_most_similar_chunks(query, chunks, embedding_model)
        if most_relevant_chunk:
            print(f"Most relevant combined chunks: {most_relevant_chunk}")
            answer = get_full_answer(most_relevant_chunk, query)
            if answer and len(answer) > 10:
                return answer

    # If no answer is found in the PDF, search Google
    print("No relevant information found in the PDF, searching Google...")
    google_snippet = search_google(query)
    if google_snippet:
        return google_snippet
    else:
        return "No relevant information found in the PDF or external sources."

if __name__ == "__main__":
    PDF_PATH = r'C:\Users\afshe\OneDrive - Kumaraguru College of Technology\Desktop\network.pdf'
    OUTPUT_TXT = "textFile.txt"
    
    text = extract_text_from_pdf(PDF_PATH)
    if text:
        with open(OUTPUT_TXT, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
        text_chunks = split_text_into_chunks(text)
        save_to_mongodb(text_chunks, embedding_model)

    user_query = "What is circuit switching?"
    response = handle_user_query(user_query, collection)
    print(f"Response: {response}")













  

#retrieve the exact answer for the query.
#handling of the queries if the query is out of pdf.
#Reduce the time to get the response from the database.











