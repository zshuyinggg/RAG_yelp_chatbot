import sys
sys.path.append('/home/zshuying/RAG_yelp_chatbot')

import utils
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain

import utils
import faiss

from langchain.chains import ConversationChain

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('RAG-based Restaurant Recommendation')
st.write('I am a RAG-based chatbot! Currently I am only referring to restaurants in Philadelphia')
st.write('Currently I am deployed on a cheap cpu server so please be patient with me as I am not very fast!')

# st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/1_%F0%9F%92%AC_basic_chatbot.py)')

class BasicChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.retriever = self.setup_retriever()
        self.question_answer_chain = self.setup_question_answer_chain()
    
    def setup_retriever(self):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vector_store = FAISS.load_local(
            folder_path='/home/zshuying/RAG_yelp_chatbot/data/embedding/faiss_index_minilm_philadelphia',
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store.as_retriever(search_kwargs={'k': 10})

    def setup_question_answer_chain(self):
        system_prompt = (
            "You are an assistant to recommend restaurants. "
            "You are only allowed to recommend restaurants that are provided to you as retrieved context. "
            "Each entry in the retrieved context is a formatted as 'RESTAURANT NAME | ADDRESS | ATTRIBUTES | CATEGORIES | ALL REVIEWS' "
            "You need to SEPARATE the contextwith '|' to extract the restaurant name, address, attributes, categories, and all reviews from the retrieved context. "
            "You need to tell the user the exact restaurant name, exact address, and summarize the attributes, categories, and all reviews for their question. "
            "DO NOT make up any restaurants that are not in the retrieved context. "
            "You can suggest multiple restaurants if needed.\n\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        return create_stuff_documents_chain(self.llm, prompt)

    @utils.enable_chat_history
    def main(self):
        # Initialize messages if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Display assistant response with streaming
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                
                try:
                    rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)
                    response = rag_chain.invoke({"input": user_query})
                    full_response = response["answer"]
                    retrieved_context = response.get("context", "No context retrieved")
                    
                    # debug
                    st.write("User Query:", user_query)
                    st.write("Retriever Configuration:", self.retriever)
                    # retrieved_docs = self.retriever.get_relevant_documents(test_query)

                    # print("\nRetrieved documents:")
                    # for i, doc in enumerate(retrieved_docs, 1):
                    #     print(f"\nDocument {i}:")
                    #     print(doc.page_content)
                    #     print("-" * 50)
                    # placeholder.markdown(f"Retrieved Context: {retrieved_context}\n\nResponse: {full_response}")
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    print(f"Error: {e}")

if __name__ == "__main__":
    obj = BasicChatbot()
    obj.main()
