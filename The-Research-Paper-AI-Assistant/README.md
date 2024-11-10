# The Academic Research Paper AI Assistant

## Developed by: Rishabh Soni, IIT BHU Varanasi

## Introduction

An academic research paper assistant designed to facilitate research by integrating Large Language Models (LLMs) with a user-friendly interface. The assistant supports streamlined search, retrieval, summarization, and analysis of academic papers, providing a multi-functional research aid for researchers. By leveraging frameworks like Streamlit and FastAPI and utilizing open-source LLMs, the application offers research assistance through a range of intelligent agents and functionalities, including future work generation.

## Features

**Technology Stack:**

- **Streamlit:** Frontend framework for creating an interactive and responsive UI.
- **FastAPI:** Backend service enabling multi-agent interactions and API functionality.
- **Neo4j:** Graph database for effective data querying and analysis.
- **Transformers/Ollama:** For LLM-powered question answering and summarization.

**Features:**

- **Summarization:** Synthesizes contributions from recent years, generating new work ideas based on recent advancements.
- **Question Answering:** Answers questions about paper content, including complex data like images and charts.
- **Review Generation:** Produces review papers highlighting state-of-the-art developments and suggesting future work directions.

**Functionality:**

- **Clustering:** Organizes text content by similarity for enhanced readability and analysis.
- **RAG Integration:** Uses retrieval-augmented generation to pull relevant information for user queries.
- **Translation and Audio:** Provides multi-language support and audio responses.

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (optional but recommended)

### Steps

1. **Clone the repository**:

   ```sh
   git clone https://github.com/rasrishabh/The-Research-Paper-AI-Assistant-.git
   ```

2. **Create a virtual environment**:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Set up your environment variables**:
   - **Pinecone API Key**: Set up your Pinecone API key as an environment variable.
   - **Other necessary keys**: Depending on your translation and TTS services, set up those API keys.

## Usage

1. **Run the Streamlit app**:

   ```sh
   streamlit run app.py
   ```

2. **Open the provided URL** in your browser to access the app.

### Prerequisites

- Python 3.8 or higher
- Virtual environment (optional but recommended)
