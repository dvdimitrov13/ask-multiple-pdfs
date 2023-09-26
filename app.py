from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from streamlit_chat import message
from agents.EY_policy_assistant_agent import EYPoliciesLookupAgent


# import langchain
# langchain.debug = True


def main():
    st.set_page_config(page_title="Your helpful AI assistant", page_icon=":books:")
    st.header('EY Internal Policies Assitant :books:')

    # Initialize the session state for chat history
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! How can I help you?"]

    # Placeholders to reduce the rerun flicker
    chat_placeholder = st.empty()
    input_placeholder = st.empty()

    # Container for chat history
    response_container = chat_placeholder.container()

    with input_placeholder:
        user_input = st.text_input("Your question!")

        if user_input:
            # Clear the text input
            st.text_input("Your question!", value="", key="reset")
            
            # Get the bot's response
            bot_response = EYPoliciesLookupAgent(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(bot_response)
        

    if st.session_state['generated']:
        with response_container:
            message(st.session_state["generated"][0], key=str(0), avatar_style='initials',seed='EY')
            for i in range(len(st.session_state['past'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style='personas',seed=55)
                message(st.session_state["generated"][i+1], key=str(i+1), avatar_style='initials',seed='EY')

if __name__ == '__main__':
    main()



