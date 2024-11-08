import utils
import streamlit as st

from langchain.chains import ConversationChain

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('RAG-based Restaurant Recommendation')
st.write('Allows users to interact with the LLM')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/1_%F0%9F%92%AC_basic_chatbot.py)')

class BasicChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
    
    def setup_chain(self):
        chain = ConversationChain(llm=self.llm, verbose=False)
        return chain
    
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
                    for chunk in self.llm.stream(user_query):
                        if hasattr(chunk, 'content'):
                            full_response += chunk.content
                            placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    print(f"Error: {e}")

if __name__ == "__main__":
    obj = BasicChatbot()
    obj.main()