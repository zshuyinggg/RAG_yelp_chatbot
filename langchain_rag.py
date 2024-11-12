from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain

import utils
import faiss


# load embeddings and index into vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
vector_store = FAISS.load_local(
            folder_path='/home/zshuying/RAG_yelp_chatbot/data/embedding/faiss_index_minilm_philadelphia',
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
retriever = vector_store.as_retriever(search_kwargs={'k': 3})

# Test retriever
test_query = "what is the best Italian restaurant in philadelphia?"
retrieved_docs = retriever.get_relevant_documents(test_query)

print("Query:", test_query)
print("\nRetrieved documents:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\nDocument {i}:")
    print(doc.page_content)
    print("-" * 50)

# Print vector store stats
print("\nVector store stats:")
print(f"Number of documents in index: {vector_store.index.ntotal}")
print(f"Embedding dimension: {vector_store.index.d}")