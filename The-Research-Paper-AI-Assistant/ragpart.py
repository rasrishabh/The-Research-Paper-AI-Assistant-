import os
import fitz  # PyMuPDF
import re
import streamlit as st
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from huggingface_hub import InferenceClient
import sys
import json
import numpy as np

# Initialize Neo4j
neo4j_uri = st.secrets["general"]["NEO4J_URI"]
neo4j_user = st.secrets["general"]["NEO4J_USER"]
neo4j_password = st.secrets["general"]["NEO4J_PASSWORD"]

class Neo4jDatabase:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            
    def create_indexes(self):
        with self.driver.session() as session:
            # Create index on Chunk nodes
            session.run("""
                CREATE INDEX chunk_id IF NOT EXISTS
                FOR (c:Chunk)
                ON (c.chunk_id)
            """)
            
    def create_vector_index(self):
        with self.driver.session() as session:
            # Create vector index for similarity search
            session.run("""
                CALL db.index.vector.createNodeIndex(
                    'embeddings',
                    'Chunk',
                    'embedding',
                    384,
                    'cosine'
                )
            """)
            
    def store_chunk(self, chunk_id, content, embedding):
        with self.driver.session() as session:
            session.run("""
                CREATE (c:Chunk {
                    chunk_id: $chunk_id,
                    content: $content,
                    embedding: $embedding
                })
            """, chunk_id=chunk_id, content=content, embedding=embedding)
            
    def query_similar_chunks(self, query_embedding, top_k=5):
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes(
                    'embeddings',
                    $top_k,
                    $query_embedding
                ) YIELD node, score
                RETURN node.content AS content, score
                ORDER BY score DESC
            """, query_embedding=query_embedding, top_k=top_k)
            
            return [(record["content"], record["score"]) for record in result]

# Model initialization
model = SentenceTransformer('all-MiniLM-L6-v2')

def initialize_database():
    db = Neo4jDatabase(neo4j_uri, neo4j_user, neo4j_password)
    db.clear_database()
    db.create_indexes()
    db.create_vector_index()
    return db

def extract_text_from_pdf(pdf_file):
    if isinstance(pdf_file, str):
        doc = fitz.open(pdf_file)
    else:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def title_based_chunking(text):
    chunks = re.split(r'(?<=\n)\s*(?=\w)', text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def section_based_chunking(text):
    sections = re.split(r'\n\s*\n', text)
    return [section.strip() for section in sections if section.strip()]

def semantic_chunking(text, max_chunk_size=512, overlap=128):
    words = text.split(' ')
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + max_chunk_size])
        if i > 0:
            previous_chunk = ' '.join(words[max(0, i - overlap):i])
            chunk = previous_chunk + ' ' + chunk
        chunks.append(chunk)
        i += max_chunk_size - overlap
    return chunks

def combined_chunking(text):
    title_chunks = title_based_chunking(text)
    final_chunks = []
    for chunk in title_chunks:
        section_chunks = section_based_chunking(chunk)
        for section_chunk in section_chunks:
            semantic_chunks = semantic_chunking(section_chunk)
            final_chunks.extend(semantic_chunks)
    return final_chunks

def store_chunks_in_neo4j(chunks, db):
    chunk_embeddings = model.encode(chunks)
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        db.store_chunk(f"chunk-{i}", chunk, embedding.tolist())

def get_relevant_chunks(query, db, top_k=5):
    query_embedding = model.encode([query])[0].tolist()
    search_results = db.query_similar_chunks(query_embedding, top_k)
    return [content for content, _ in search_results]

def generate_response_from_chunks(chunks, query):
    combined_content = "\n".join([f"Chunk:\n{chunk}" for chunk in chunks])
    prompt_template = (
        "You are an AI research assistant. Your job is to help users understand and extract key insights from research papers. "
        "You will be given a query and context from multiple research papers. Based on this information, provide accurate, concise, and helpful responses. "
        "Here is the context from the research papers and the user's query:\n\n"
        "Context:\n{context}\n\n"
        "Query: {query}\n\n"
        "Please provide a detailed and informative response based on the given context. Make sure your response is complete and ends with 'End of response.'."
    )
    user_query = prompt_template.format(context=combined_content, query=query)
    
    huggingface_token = st.secrets["general"]["HUGGINGFACE_TOKEN"]
    client = InferenceClient("Alpha-VLLM/LLaMA2-Accessory", token=huggingface_token)

    response = client.chat_completion(
        messages=[{"role": "user", "content": user_query}],
        max_tokens=500,
        stream=False
    )

    if response and 'choices' in response and response['choices']:
        content = response['choices'][0]['message']['content']
        if 'End of response.' in content:
            content = content.split('End of response.')[0].strip()
        return content
    return "No response received."

def process_pdfs(pdf_files, query):
    db = initialize_database()
    nested_texts = []

    try:
        # Extract and clean text from each PDF
        for pdf_file in pdf_files:
            text = extract_text_from_pdf(pdf_file)
            cleaned_text = clean_text(text)
            nested_texts.append(cleaned_text)

        # Process each text and store in Neo4j
        for text in nested_texts:
            chunks = combined_chunking(text)
            store_chunks_in_neo4j(chunks, db)

        # Get relevant chunks and generate response
        relevant_chunks = get_relevant_chunks(query, db)
        response = generate_response_from_chunks(relevant_chunks, query)
        return response
    finally:
        db.close()