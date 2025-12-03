import os
import fitz  # PyMuPDF
import re
import streamlit as st
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import google.generativeai as genai
import sys
import numpy as np

# --- CONFIGURATION ---
# Initialize Neo4j
neo4j_uri = st.secrets["general"]["NEO4J_URI"]
neo4j_user = st.secrets["general"]["NEO4J_USER"]
neo4j_password = st.secrets["general"]["NEO4J_PASSWORD"]

# Initialize Gemini
if "GOOGLE_API_KEY" in st.secrets["general"]:
    genai.configure(api_key=st.secrets["general"]["GOOGLE_API_KEY"])

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
            session.run("CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)")
            
    def create_vector_index(self):
        with self.driver.session() as session:
            try:
                # We attempt to create the index
                session.run("""
                    CALL db.index.vector.createNodeIndex(
                        'embeddings', 'Chunk', 'embedding', 384, 'cosine'
                    )
                """)
            except Exception as e:
                # If the error is "EquivalentSchemaRuleAlreadyExistsException", 
                # it means the index exists. We ignore it and proceed.
                if "EquivalentSchemaRuleAlreadyExistsException" in str(e):
                    pass 
                else:
                    # If it's a different error, we raise it so you can see it
                    raise e
            
    def store_chunk(self, chunk_id, content, embedding):
        with self.driver.session() as session:
            session.run("""
                CREATE (c:Chunk {chunk_id: $chunk_id, content: $content, embedding: $embedding})
            """, chunk_id=chunk_id, content=content, embedding=embedding)
            
    def query_similar_chunks(self, query_embedding, top_k=5):
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('embeddings', $top_k, $query_embedding) 
                YIELD node, score
                RETURN node.content AS content, score
                ORDER BY score DESC
            """, query_embedding=query_embedding, top_k=top_k)
            return [(record["content"], record["score"]) for record in result]

# Use local embeddings (free and fast)
model = SentenceTransformer('all-MiniLM-L6-v2')

def initialize_database():
    try:
        db = Neo4jDatabase(neo4j_uri, neo4j_user, neo4j_password)
        db.clear_database()
        db.create_indexes()
        db.create_vector_index()
        return db
    except Exception as e:
        st.error(f"Neo4j Connection Error: {e}")
        return None

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
    return re.sub(r'\s+', ' ', text).strip()

def combined_chunking(text):
    # Simplified chunking for stability
    words = text.split(' ')
    chunks = []
    chunk_size = 300
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def store_chunks_in_neo4j(chunks, db):
    chunk_embeddings = model.encode(chunks)
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        db.store_chunk(f"chunk-{i}", chunk, embedding.tolist())

def get_relevant_chunks(query, db, top_k=5):
    if not db: return []
    query_embedding = model.encode([query])[0].tolist()
    search_results = db.query_similar_chunks(query_embedding, top_k)
    return [content for content, _ in search_results]

def generate_response_from_chunks(chunks, query):
    if not chunks:
        return "No relevant information found in the document."
        
    context = "\n".join([f"Chunk: {c}" for c in chunks])
    prompt = f"""You are a research assistant. Answer the query based ONLY on the context below.

    
    Context:
    {context}
    
    Query: {query}
    """
    
    try:
        # Switch to Flash (Faster & Free-tier friendly)
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"
