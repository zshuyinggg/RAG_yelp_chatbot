import sys
sys.path.append('/home/zshuying/RAG_yelp_chatbot')
import pandas as pd
import utils
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

import utils
import faiss

from langchain.chains import ConversationChain
from langchain.llms.base import LLM
from typing import Any, List, Optional
import requests

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('RAG-based Restaurant Recommendation')
st.write('I am a RAG-based chatbot! ')
st.write('For now, as I am still in the development stage, I only recommend restaurants in Philadelphia which has the most reviews on Yelp')

# st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/1_%F0%9F%92%AC_basic_chatbot.py)')

class HyperbolicLLM(LLM):
    api_key: str
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    temperature: float = 0.7
    top_p: float = 0.9
    
    @property
    def _llm_type(self) -> str:
        return "hyperbolic"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": self.model,
            "max_tokens": 16253,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        
        response = requests.post(
            "https://api.hyperbolic.xyz/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise ValueError(f"API request failed: {response.text}")
            
        return response.json()['choices'][0]['message']['content']

class BasicChatbot:

    def __init__(self):
        utils.sync_st_session()
        
        # Initialize the Hyperbolic LLM
        api_key = st.secrets["HYPERBOLIC_API_KEY"]  # Store your API key in .streamlit/secrets.toml
        self.llm = HyperbolicLLM(
            api_key=api_key,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            temperature=0.7,
            top_p=0.9
        )
        
        print(f"\nLLM configuration:\n{self.llm}")
        self.retriever = self.setup_retriever(k=2)
        self.question_answer_chain = self.setup_question_answer_chain()
        self.reference_df = pd.read_csv('/home/zshuying/RAG_yelp_chatbot/data/merge_businessinfo_reviews.csv')
    
    def setup_retriever(self,k=2):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vector_store = FAISS.load_local(
            folder_path='/home/zshuying/RAG_yelp_chatbot/data/embedding/faiss_index_minilm_philadelphia',
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store.as_retriever(search_kwargs={'k': k})


    def get_retrieved_docs(self, query, verbose=False):
        retrieved_docs = self.retriever.get_relevant_documents(query)

        print("Query:", query)
        print("\nRetrieved Restaurants:")
        if verbose:
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"\nRestaurant {doc.metadata['source']}:")
                print(self.reference_df[self.reference_df['business_id'] == doc.metadata['source']])
                print("-" * 50)
        return self.reference_df[self.reference_df['business_id'].isin([doc.metadata['source'] for doc in retrieved_docs])]


    def turn_retrieved_docs_into_string(self, retrieved_df):
        docs = []
        for _, row in retrieved_df.iterrows():
            # Truncate reviews to a reasonable length (e.g., first 500 characters)
            truncated_reviews = row['all_reviews'][:1000] + "..." if len(row['all_reviews']) > 1000 else row['all_reviews']
            
            content = (
                f"Restaurant Name: {row['name']}\n"
                f"Address: {row['address']}\n" 
                f"Attributes: {row['attributes']}\n"
                f"Categories: {row['categories']}\n"
                f"Key Reviews: {truncated_reviews}"
            )
            docs.append(Document(page_content=content))
        return docs



    def setup_question_answer_chain(self):
        system_prompt = (
            "You are a Philadelphia restaurant recommendation assistant. "
            "Using ONLY the restaurant information below:\n"
            "---\n"
            "{context}\n"
            "---\n"
            "Provide recommendations for restaurant that matches the request.\n"
            "Please provide the restaurant name, address and explain why it matches the request.\n"
            "You can recommend up to 3 restaurants - if fewer restaurants match the request, only include those that are relevant."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        return create_stuff_documents_chain(
            self.llm,
            prompt,
            document_variable_name="context"
        )

    @utils.enable_chat_history
    def main(self):
        # Initialize messages if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Add text input for user query at bottom
        user_input = st.text_input("Enter your restaurant query here:", placeholder="e.g. Italian restaurants with outdoor seating")
        
        if user_input:  # Use the text input instead of chat input
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display assistant response with streaming
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                
                try:
                    # Get retrieved documents and convert to string context
                    retrieved_df = self.get_retrieved_docs(user_input, verbose=True)
                    context = self.turn_retrieved_docs_into_string(retrieved_df)
                    
                    # Add debug prints
                    print("\nContext being passed to QA chain:")
                    for doc in context:
                        print(f"\nDocument content:\n{doc.page_content}\n---")
                    
                    response = self.question_answer_chain.invoke({
                        "context": context,
                        "input": user_input
                    })
                    
                    full_response = response
                    placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    print(f"Error: {e}")

if __name__ == "__main__":
    obj = BasicChatbot()
    obj.main()
