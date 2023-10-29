from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import json
from streamlit_chat import message
from langchain.load.dump import dumps
from agents.EY_policy_assistant_agent import EYPoliciesLookupAgentClass


# import langchain
# langchain.debug = True


def main():
    st.set_page_config(page_title="Your helpful AI assistant", page_icon=":books:")
    st.header('EY Internal Policies Assistant :books:')

    # Initialize the session state for chat history
    if 'agent' not in st.session_state:
        st.session_state['agent'] = EYPoliciesLookupAgentClass()
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! How can I help you?"]

    # Placeholders to reduce the rerun flicker
    chat_placeholder = st.empty()
    input_placeholder = st.empty()

    # Container for chat history
    response_container = chat_placeholder.container()

    with input_placeholder.container():
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("Ask me a question:", key='input', height=50)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            # Get the bot's response
            bot_response = st.session_state['agent'].run(user_input)

            # Extract and pretty-print the intermediate steps
            intermediate_steps = bot_response["intermediate_steps"]

            bot_reply = ''

            for step, observation in intermediate_steps:
                bot_reply += step.log


            bot_reply += 'Final Answer: \n' + bot_response['output']

            # Append the user's question to the chat history
            st.session_state['past'].append(user_input)

            # Append the intermediate steps and the final reply to the chat history
            st.session_state['generated'].append(bot_reply)

    if st.session_state['generated']:
        with response_container:
            message(st.session_state["generated"][0], key=str(0), avatar_style='initials',seed='EY')
            for i in range(len(st.session_state['past'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style='personas',seed=55)
                message(st.session_state["generated"][i+1], key=str(i+1), avatar_style='initials',seed='EY')

if __name__ == '__main__':
    main()



