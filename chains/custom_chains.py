from langchain.vectorstores.base import VectorStoreRetriever
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.chat_models import AzureChatOpenAI
from langchain.llms import AzureOpenAI

from templates import chain_templates

llm = AzureChatOpenAI(deployment_name="gpt-4", model_name="gpt-4", temperature=0)
llm_creative = AzureChatOpenAI(
    deployment_name="gpt-4", model_name="gpt-4", temperature=1
)

verbose = True


def get_condense_question_chain():
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
        chain_templates.condense_question
    )

    return LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=verbose)


def get_document_map_reduce_chain():
    COMBINE_PROMPT = PromptTemplate(
        template=chain_templates.documents_combine,
        input_variables=["summaries", "question"],
    )

    DOC_PROMPT = PromptTemplate(
        template=chain_templates.document_with_source,
        input_variables=["page_content", "source"],
    )

    QUESTION_PROMPT = PromptTemplate(
        template=chain_templates.document_reduce,
        input_variables=["context", "question"],
    )

    return load_qa_with_sources_chain(
        llm,
        chain_type="map_reduce",
        document_prompt=DOC_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        question_prompt=QUESTION_PROMPT,
        verbose=verbose,
    )


def get_conversational_retrieval_chain(vectorstore, memory, retrieved_docs=5):

    return ConversationalRetrievalChain(
        retriever=VectorStoreRetriever(vectorstore=vectorstore, search_kwargs={"k": retrieved_docs}),
        memory=ReadOnlySharedMemory(memory=memory),
        question_generator=get_condense_question_chain(),
        combine_docs_chain=get_document_map_reduce_chain(),
        verbose=verbose,
    )
