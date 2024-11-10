import requests
import feedparser
import os
import fitz  # PyMuPDF
import re
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import zipfile
import io

nltk.download('punkt_tab')
nltk.download('stopwords')



# Sanitizes a string to be used as a valid filename on Windows
def sanitize_filename(filename):
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.strip().strip('.').replace('\n', ' ').replace('\r', '')
    return filename[:15]

# Searches ArXiv for papers based on the query and retrieves metadata
def search_arxiv(query, max_results=10):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=all:{query}&start=0&max_results={max_results}"
    url = base_url + search_query
    response = requests.get(url)
    feed = feedparser.parse(response.content)
    
    results = []
    for entry in feed.entries:
        result = {
            'title': entry.title,
            'summary': entry.summary,
            'authors': [author.name for author in entry.authors],
            'published': entry.published,
            'link': entry.link,
            'pdf_link': entry.link.replace("abs", "pdf") + ".pdf"
        }
        results.append(result)
    
    return results

# Downloads a PDF file from the provided URL and saves it
def download_pdf(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

# Extracts text content from a PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Extracts images from a PDF using PyMuPDF and saves them to the specified directory
def extract_images_from_pdf(pdf_path, save_dir):
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_name = f"image_{page_num}_{img_index}.png"
            image_path = os.path.join(save_dir, image_name)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            images.append(image_path)

    return images

# Reads and extracts text from a single PDF file
def read_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        page_text_all = ""
        for page_number in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_number]
            page_text_all += page.extract_text()
        return page_text_all

# Reads and extracts text from multiple PDF files
def read_multiple_pdfs(pdf_paths):
    dfs = []
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file '{pdf_path}' not found. Skipping...")
            continue
        
        pdf_text = read_pdf(pdf_path)
        df = pd.DataFrame({'PDF File': [pdf_path], 'Text': [pdf_text]})
        dfs.append(df)
    
    if not dfs:
        print("No valid PDF files found. Exiting.")
        return None
    
    result_df = pd.concat(dfs, ignore_index=True)
    return result_df


def download_pdfs_as_zip(urls):
    """Download PDFs from URLs and zip them."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, 'w') as zip_file:
        for url in urls:
            response = requests.get(url)
            filename = url.split("/")[-1]
            zip_file.writestr(filename, response.content)

    buffer.seek(0)
    return buffer

def process_docs(selected_indices, arxiv_results, save_dir):
    pdf_paths = []
    for selection in selected_indices:
        if 0 <= selection < len(arxiv_results):
            selected_result = arxiv_results[selection]
            pdf_url = selected_result['pdf_link']
            paper_title = selected_result['title']
            
            # Sanitize the paper title for filename
            sanitized_title = sanitize_filename(paper_title)
            
            save_path = os.path.join(save_dir, f"{sanitized_title}.pdf")
            
            print(f"Downloading PDF from: {pdf_url}")
            download_pdf(pdf_url, save_path)
            print(f"PDF downloaded and saved to {save_path}")
            pdf_paths.append(save_path)
        else:
            print(f"Invalid selection {selection + 1}. Skipping.")
    
    # Read and process selected PDFs
    result_df = read_multiple_pdfs(pdf_paths)
    
    if result_df is None:
        exit(1)

    processed_documents = []
    for document in result_df['Text']:
        document = document.lower()
        tokens = word_tokenize(document)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        filtered_tokens = [word for word in filtered_tokens if word.isalpha() or (word.isalnum() and not word.isnumeric())]
        filtered_tokens = np.char.replace(filtered_tokens, "'", "")
        new_text = " ".join([w for w in filtered_tokens if len(w) > 1])
        processed_documents.append(new_text)

    return result_df, processed_documents


def process_docs2(selected_indices, arxiv_results):
    pdf_urls = []
    for selection in selected_indices:
        if 0 <= selection < len(arxiv_results):
            selected_result = arxiv_results[selection]
            url = selected_result['pdf_link']
            pdf_urls.append(url)
        else:
            print(f"Invalid selection {selection + 1}. Skipping.")
    
    buffer = download_pdfs_as_zip(pdf_urls)
    
    return buffer

def text_from_file_uploader(uploaded_files):
    pdf_text = []
    for uploaded_file in uploaded_files:
        # Read the PDF file
        text=''
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf_document:
            # Extract text from each page
            for page_number in range(len(pdf_document)):
                page = pdf_document.load_page(page_number)
                text += page.get_text()
        pdf_text.append({"name": uploaded_file.name, "text": text})
    return pdf_text

def tokenize_text(pdf_text):
    processed_documents = []
    for document in pdf_text:
        document = document['text'].lower()
        tokens = word_tokenize(document)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        filtered_tokens = [word for word in filtered_tokens if word.isalpha() or (word.isalnum() and not word.isnumeric())]
        filtered_tokens = np.char.replace(filtered_tokens, "'", "")
        new_text = " ".join([w for w in filtered_tokens if len(w) > 1])
        processed_documents.append(new_text)
    return processed_documents



def clustering(result_df, processed_documents):
    tfidf = TfidfVectorizer()
    response = tfidf.fit_transform(processed_documents)
    feature_names = tfidf.get_feature_names_out()

    # Determine the optimal number of clusters using silhouette score
    silhouette_scores = []
    k_range = range(2, 10)

    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(response)
            score = silhouette_score(response, kmeans.labels_)
        except ValueError:
            return result_df, "Error"
        else:
            silhouette_scores.append(score)

    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"Optimal number of clusters: {optimal_k}")


    # Cluster with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(response)

    # Add cluster labels to the DataFrame
    result_df = pd.DataFrame(result_df)

    result_df['Cluster'] = kmeans.labels_
    
    # Identify the top keywords for each cluster
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    keywords = []
    for i in range(optimal_k):
        cluster_keywords = [feature_names[ind] for ind in order_centroids[i, :10]]
        keywords.append(cluster_keywords)
    
    cluster_centroids = [" ".join(keywords[i]) for i in range(optimal_k)]
    
    # Add the cluster centroids to the DataFrame
    cluster_centroid_mapping = {i: cluster_centroids[i] for i in range(optimal_k)}
    result_df['Cluster Keywords'] = result_df['Cluster'].map(cluster_centroid_mapping)

    # Visualize the clusters
    # Reduce the TF-IDF feature space to 2D for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(response.toarray())
    reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)

    fig=plt.figure(figsize=(12, 8))
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=result_df['Cluster'], palette='viridis')
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], s=300, c='red', marker='X')
    for i, cluster_center in enumerate(reduced_cluster_centers):
        plt.text(cluster_center[0], cluster_center[1], f'Cluster {i}: {cluster_centroids[i]}', fontsize=12, ha='right')
    plt.title('Document Clusters with Centroid Keywords')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()

    return result_df, fig
    
import os
import glob
import PyPDF2

def list_pdfs(directory):
    # Use glob to find all PDFs in the given directory
    pdf_files = glob.glob(os.path.join(directory, '*.pdf'))
    return pdf_files
