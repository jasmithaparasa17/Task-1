# Task-1
Chat With PDF Using RAG Pipeline
# Install required libraries
!pip install PyPDF2 langchain sentence-transformers faiss-cpu openai
!pip install -U langchain-community

# Import necessary libraries
import os
import PyPDF2
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

#Loading the PDF File
pdf_path = "/content/Tables- Charts- and Graphs with Examples from History- Economics- Education- Psychology- Urban Affairs and Everyday Life - 2017-2018.pdf"

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Step 2: Chunk the Text
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Step 3: Generate Embeddings
def create_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings, chunks

# Step 4: Store Embeddings in Vector Database
def create_vector_database(embeddings, chunks):
    return FAISS.from_texts(chunks, embeddings)

"""def handle_query(query, retriever, llm):
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)  """
from transformers import pipeline

#Handle Queries Using QA Pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def handle_query(query, retriever): 
    docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in docs])
    answer = qa_pipeline(question=query, context=context)
    return answer["answer"]

# Main Code Execution
if __name__ == "__main__":
    # Extract and process text
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    # Generate embeddings and create a vector database
    embeddings, processed_chunks = create_embeddings(chunks)
    vector_db = create_vector_database(embeddings, processed_chunks)

    # Create the LLM and connect it with the retriever
    llm = OpenAI(api_key="your_openai_api_key")  
    retriever = vector_db.as_retriever()

    # Example Query
    query = "What is the unemployment information based on degree type?"
    response = handle_query(query, retriever)
    print("Response:", response)
