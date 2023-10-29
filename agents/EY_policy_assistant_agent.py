import os
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import AzureChatOpenAI
from langchain.load.dump import dumps
from langchain.memory import ConversationBufferMemory

from chains.custom_chains import get_conversational_retrieval_chain
from utils import process_input, vectorstores

from langchain.chains import LLMChain
from langchain.agents.chat.base import ChatAgent
from typing import Any, List, Optional, Sequence
from langchain.agents.chat.prompt import (
    FORMAT_INSTRUCTIONS,
    HUMAN_MESSAGE,
    SYSTEM_MESSAGE_PREFIX,
    SYSTEM_MESSAGE_SUFFIX,
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import Field
from langchain.schema import BasePromptTemplate
from langchain.tools.base import BaseTool

import langchain

langchain.debug


_ROLE_MAP = {"human": "Human: ", "ai": "Assistant: "}


@classmethod
def create_prompt(
    cls,
    tools: Sequence[BaseTool],
    system_message_prefix: str = SYSTEM_MESSAGE_PREFIX,
    system_message_suffix: str = SYSTEM_MESSAGE_SUFFIX,
    human_message: str = HUMAN_MESSAGE,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
) -> BasePromptTemplate:
    tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    format_instructions = format_instructions.format(tool_names=tool_names)
    template = "\n\n".join(
        [
            system_message_prefix,
            tool_strings,
            format_instructions,
            system_message_suffix,
        ]
    )

    ## Custom messages template, needs adjusting some issues but shows potential

    ## Investigate where does the tool description appear
    messages = [
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(human_message),
    ]
    if input_variables is None:
        input_variables = ["input", "chat_history", "agent_scratchpad"]
    return ChatPromptTemplate(input_variables=input_variables, messages=messages)

# Modify the prompt creatiopn of the CHAT_ZERO_SHOT_REACT_DESCRIPTION to include memory through monkey pathching
ChatAgent.create_prompt = create_prompt


class EYPoliciesLookupAgentClass:
    def __init__(self):
        self.SYSTEM_MESSAGE_PREFIX = """You are a large language model serving as EY's internal policies assistant. 
You are designed to assist with inquiries related exclusively to EY's policies and guidelines but should still interact with the Human in a natural and conversational way.
You are committed to providing truthful answers; if there's no available information on a topic, the response will candidly reflect that absence of data. 
You must give comprehensive and exact responses including all detail from the retrieved document summaries, always cite the document sources on a new line at the end of your message. Answers must remain confined to EY's policies and guidelines. The Assistant is equipped with specific tools to facilitate this focus. Remember, the primary goal is accuracy and transparency in addressing all inquiries related to EY's policies.
Answer the following questions as best you can. You have access to the following tools:"""

        self.SYSTEM_MESSAGE_SUFFIX = ""

        self.HUMAN_MESSAGE = """Here is my input! Reminder to always use the exact combination of
Thought: string // Your thoughts about my input here
Final Answer: string // What you want me to see here
when you want to provide either your final answer or you want to adress me directly!

Here is my input and the steps you have taken so far:
{input}\n\n{agent_scratchpad}"""

        self.llm = AzureChatOpenAI(
            deployment_name="gpt-4", model_name="gpt-4", temperature=0
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="output", return_messages=True
        )
        pdf_docs = [
            os.path.join("./EYPolicies/", filename)
            for filename in os.listdir("./EYPolicies/")
            if filename.endswith(".pdf")
        ]
        text_chunks = process_input.process_pdf(pdf_docs)
        self.vectorstore = vectorstores.get_FAISS(text_chunks)
        self.chain = get_conversational_retrieval_chain(self.vectorstore, self.memory)
        self.tools = [
            Tool(
                name="EY-internal-policies",
                func=self.chain.run,
                description="useful for quering EY internal policies and guidelines. ALWAYS ask one question related to one topic at a time in order to ensure maximum detail in the response. All information returned by the tool should be used and ALWAYS include the sources at the end of the Final Answer!",
            )
        ]
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message_prefix": self.SYSTEM_MESSAGE_PREFIX,
                "system_message_suffix": self.SYSTEM_MESSAGE_SUFFIX,
                "human_message": self.HUMAN_MESSAGE
            },
            return_intermediate_steps=True,
            max_iterations=5,
        )

    def run(self, question: str) -> str:
        response = self.agent({"input": question})
        return response
