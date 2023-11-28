import os
import requests
#import langchain
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st


# for installing this these libs you have to off the proxy and then install it
#install transformer lib
#install numpy lib
#install pdfplumer

# for running this application you guys have to set proxy and then only it work as ford llm api will work with proxy only



token_endpoint = "Enter Your Token End Point"
api_host = "api_host"
api_endpoint = "api_endPoint"
proxy_endpoint = "Porxy_endPoint"
scope = "Scope_point"
client_id = "Enter the Client ID"
client_secret = "Enter Client Secret"


os.environ['HTTP_PROXY'] = proxy_endpoint
os.environ['HTTPS_PROXY'] = proxy_endpoint
os.environ['NO_PROXY'] = api_host




# Load a pre-trained sentence embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# List of PDF file paths
pdf_files = ["Ford OS Behaviors and Anchors.pdf", "IntroductiontoWasteManagementtextbookpub.pdf", "2023 World Mental Health Day.pdf"]   # Add your file paths

# Query
#query = "tell me about waste management how the waste management is done"

# Limit for token size
max_token_size = 1024


def relevent_context(query):
  
  # List to store relevant information
    relevant_document_data = []

      # Collect relevant information
    for pdf_file in pdf_files:
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_file)
        
        # Encode the entire text
        embedding = model.encode(text)
        
        # Limit token size for the relevant information based on the query
        query_tokens = query.split()
        if any(token in text for token in query_tokens):
            relevant_text = " ".join(text.split()[:max_token_size])
            relevant_document_data.append({'text': relevant_text, 'embedding': embedding})

    # Convert the list of embeddings to a NumPy array
    relevant_document_embeddings_np = np.array([data['embedding'] for data in relevant_document_data])

    # Perform similarity search using Faiss
    index = faiss.IndexFlatL2(len(relevant_document_embeddings_np[0]))
    index.add(relevant_document_embeddings_np)

    # Embedding for the query
    query_embedding = model.encode(query)

    # Number of nearest neighbors to search for
    k = 1

    # Search for the nearest neighbors
    distances, indices = index.search(query_embedding.reshape(1, -1), k)


    # Print the results
    #print("Query:")
    #print(query)
    #print("\nIndices of Nearest Neighbors:")
    #print(indices)
    #print("\nDistances to Nearest Neighbors:")
    #print(distances)

    # Extract the relevant text for each nearest neighbor and limit token size

    val=""
    for i in indices.flatten():
        print("\nRelevant Text for Neighbor", i)
        val+=relevant_document_data[i]['text']
        # print(relevant_document_data[i]['text'])
    
    return val

token = None

def get_token():
    response = requests.post(token_endpoint, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
        "grant_type": "client_credentials"
    })

    return response.json()['access_token']


token = get_token()


def call_api(token,query, relative_ans):
    response = requests.post(api_endpoint,
    headers={
        "Authorization": f"Bearer {token}"
    },
    json={
        "model":"gpt",
        "context": "You are a helpful assistant" + query,
        "messages": [{
            "role": "user",
            "content":relative_ans+" if you don't know then say you don't know the answer"
        }],
    })

    #print(response.json())
    
   
    res=response.json()['content']
    with st.container():
      st.write(res)
    
    return 




if not token:
    print('Cannot aquire token.')
    exit()

#response = call_api(token,query)
#print(response)



# Define the Streamlit app for User Interface


st.title('Ford Text_Summarizer App')

# Create a form for uploading files and entering text
#with st.form('my_form'):
    #uploaded_files = st.file_uploader('Upload your files:', type='pdf', accept_multiple_files=True)
    #text = st.text_area('Enter text:', '')
    #submitted = st.form_submit_button('Submit')

# Create a form for entering text
with st.form('my_form'):
    text = st.text_area('Enter text:', '')
    submitted = st.form_submit_button('Summarize')

    # Run the chat_TBEA function when the form is submitted
    if submitted:
        relative_ans=relevent_context(text)
        call_api(token,text,relative_ans)

