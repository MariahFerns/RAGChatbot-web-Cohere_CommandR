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


# +
# Define the main function to generate responses

def generate_rag_response_with_citations(query):
    '''
    Generates a response to the user query using Cohere's Command-R LLM with inbuilt RAG capability 
    by referencing a set of user uploaded documents and includes citations in the response.
    
    Parameters:
    - query (str): This is the query entered by the user.
    
    Output:
    - Tuple (str, list): This includes the generated response and a list of citations
    '''
      
    # Calling the Cohere chat endpoint
    response = co.chat(
        model = 'command-r',
        message = query,
        connectors = [{'id': 'web-search'}] # Using the web search connector
    )
    
    # Extract text and citations from response
    text = response.text
    citations = response.citations
    
    # Return the output
    return text, citations


# +
# Building the Streamlit UI

st.title('🔖RAG with Citations using Command-R : Web Search')

# Get input parameters
user_query = st.text_area('Enter your query:')

# Getting the response and validations on button click
with st.form('summarize_form', clear_on_submit=True):
    cohere_api_key = st.text_input('Cohere API Key', type='password')
    submitted = st.form_submit_button('Submit')
    
    if submitted and len(cohere_api_key):
        if st.button('Get answer'):
            if not user_query: # if user_query is empty
            st.write('Please enter a query to proceed.')
        else:
            # Call the main function to get responses & citations
            with st.spinner('Getting results...'):
                response, citations = generate_rag_response_with_citations(user_query)
                del cohere_api_key
                
                # Print the response
                st.write('Answer:')
                st.write(response)

                # Print the citations (if any)
                if citations:
                    st.write('Citations:')
                    for citation in citations:
                        cited_text = citation.text
                        # Display cited text and a notice that it is sourced from the web
                        st.write(f'- {cited_text} (source from web search)')
                
