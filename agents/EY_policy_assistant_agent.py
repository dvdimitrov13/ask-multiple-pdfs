import os
from langchain.load.dump import dumps
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import AzureChatOpenAI

from langchain import PromptTemplate

from utils import process_input, vectorstores
from chains.custom_chains import get_conversational_retrieval_chain


def EYPoliciesLookupAgent(question: str) -> str:

    PREFIX = """EY's Internal Policies Assistant is a large language model. It's designed to assist with inquiries related exclusively to EY's policies and guidelines. As a language model, Assistant strives to generate human-like text based on the input it receives, ensuring that the conversations sound natural and the responses are relevant. The Assistant is committed to providing truthful answers; if there's no available information on a topic, the response will candidly reflect that absence of data. 

You are currently interacting with EY's Internal Policies Assistant. It is imperative to answer questions as comprehensively as possible and to cite sources when they are accessible. Answers must remain confined to EY's policies and guidelines. The Assistant is equipped with specific tools to facilitate this focus. Remember, the primary goal is accuracy and transparency in addressing all inquiries related to EY's policies.
"""


    llm = AzureChatOpenAI(deployment_name="gpt-4", model_name="gpt-4", temperature=0.5)

    memory = ConversationBufferMemory(memory_key="chat_history", output_key='output', return_messages=True)

    pdf_docs = [
        os.path.join("./EYPolicies/", filename)
        for filename in os.listdir("./EYPolicies/")
        if filename.endswith(".pdf")
    ]

    ## This can be optimized
    text_chunks = process_input.process_pdf(pdf_docs)
    vectorstore = vectorstores.get_FAISS(text_chunks)
    chain = get_conversational_retrieval_chain(vectorstore, memory)

    tools = [
        Tool(
            name="EY-internal-policies",
            func=chain.run,
            description="useful for quering EY internal policies and guidelines, returns a summary of documents relevant to the query with sources, query should be formatted as a question",
        )
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": PREFIX},  # here
        return_intermediate_steps=True,
    )

    response = agent({'input': question})
    return response['chat_history'][-1].content
