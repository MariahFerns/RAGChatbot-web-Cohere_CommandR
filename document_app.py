# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Install required libraries
# # !pip install -r requirements.txt
# # !pip install python-dotenv

# +
# Import required libraries

import streamlit as st
import cohere
import os
from dotenv import load_dotenv

# +
# Getting API key from the environment variable

load_dotenv()
api_key = os.getenv('COHERE1_API_KEY')
print(f'API Key: {api_key}')

# +
# Initializing the Cohere client with API key

if api_key is None:
    st.error('COHERE_API_KEY environment variable is not found. Please check if it is set.')
else:
    co = cohere.Client(api_key)


# +
# Define the main function to generate responses

def generate_rag_response_with_citations(query, documents):
    '''
    Generates a response to the user query using Cohere's Command-R LLM with inbuilt RAG capability 
    by referencing a set of user uploaded documents and includes citations in the response.
    
    Parameters:
    - query (str): This is the query entered by the user.
    - documents (list): This is a list of documents provided by the user.
    
    Output:
    - Tuple (str, list): This includes the generated response and a list of citations
    '''
    # Format documents for API
    formatted_documents = [ {'title': f'doc_{i}', 'snippet': doc} for i, doc in enumerate(documents)]
    
    # Calling the Cohere chat endpoint
    response = co.chat(
        model = 'command-r',
        message = query,
        documents = formatted_documents
    )
    
    # Extract text and citations from response
    text = response.text
    citations = response.citations
    
    # Return the output
    return text, citations


# +
# Building the Streamlit UI

st.title('🔖RAG with Citations using Command-R : Document Search')

# Get input parameters
uploaded_files = st.file_uploader('Upload documents related to your query (text files):', accept_multiple_files=True, type=['txt'])
user_query = st.text_area('Enter your query:')

# Getting the response and validations on button click
if st.button('Get answer'):
    if not user_query: # if user_query is empty
        st.write('Please enter a query to proceed.')
    elif not uploaded_files: # if no documents are uploaded
        st.write('Please upload atleast one document to proceed.')
    else:
        # Read content of uploaded files, decodes to string
        documents = [ file.getvalue().decode('utf-8') for file in uploaded_files]
        
        # Call the main function to get responses & citations
        response, citations = generate_rag_response_with_citations(user_query, documents)
        
        # Print the response
        st.write('Answer:')
        st.write(response)
        
        # Print the citations (if any)
        if citations:
            st.write('Citations:')
            for citation in citations:
                cited_text = citation.text
                document_ids = citation.document_ids
                # As document_ids are in the format doc_i, extract and display the cited document snippet
                for doc_id in document_ids:
                    index = int(doc_id.split('_')[-1])
                    st.write(f'- {cited_text} (from document: {documents[index]})')

# -


