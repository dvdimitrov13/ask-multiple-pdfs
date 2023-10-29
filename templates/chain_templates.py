condense_question = """Given the following conversation and a follow up query, rephrase the follow up query into a short description of the information that needs to be found.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone query:"""

document_reduce = """Use the following portion of a long document to determine if any part of it is potentially relevant to the query. Summarize only the relevant information succinctly without loss of information. 
{context}
Query: {question}
Summary of relevant information, if any:"""

document_with_source='''Summary: {page_content}
Source: {source}'''

documents_combine = """Given the following summaries of extracted text from retrieved documents and a query, create a final answer with references ("SOURCES").
If a source has no rlevant information the summary will state so, base your answer on summaries from relevant sources, if there are none say that no relevant information could be retrieved.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part as a newline at the end of your answer.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER: Here is all I came up with!"""



