from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file

from haystack import Document, Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

from haystack.dataclasses import ChatMessage

documents = [
    Document(content="Mark lives in Berlin"),
    Document(content="Jean lives in Paris"),
    Document(content="Mark speaks Turkish"),
    Document(content="Jean was born in Belgium"),
    ]

# Add documents to the document_store
document_store = InMemoryDocumentStore()
document_store.write_documents(documents)

# Create the retriever with the document_store
retriever = InMemoryBM25Retriever(document_store)


# Create prompt builder component
prompt_template = """
Given these documents, answer the question.
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
"""
prompt_builder = PromptBuilder(template=prompt_template)

# Create generator component
model_name = "gpt-4o-mini"
llm = OpenAIGenerator(model=model_name)

# Build pipeline and create connections between components
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("generator", llm)

rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "generator.prompt")

from pathlib import Path
rag_pipeline.draw(Path("pipeline.png"))

# Run pipeline
questions = [
    "Where does Mark live?",
    "What language does Mark speak?",
    "Where was Jean born?",
    "Where does Jean live?",
]

for query in questions:
    result = rag_pipeline.run({"retriever": {"query": query}, "prompt_builder": {"question": query}})
    llm_response = result["generator"]["replies"][0]

    print(f"User: {query}")
    print(f"Assistant: {llm_response}\n")


# Create a new, complete chatbot pipeline with memory
# We need to create each component again in order to be able to use them in a new pipeline

# Create the retriever with the document_store
retriever = InMemoryBM25Retriever(document_store)

# Create prompt builder component
prompt_template = """
Given these documents and the current conversation, answer the question.
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}

Below is the current conversation consisting of interleaving human and assistant messages.
Current Conversation:
{% for chat in chat_history %}
    {{ chat.role.name }}:  {{ chat.content }}
{% endfor %}

Answer:
"""
prompt_builder = PromptBuilder(template=prompt_template)

# Create generator component
model_name = "gpt-4o-mini"
llm = OpenAIGenerator(model=model_name)

# Build pipeline and create connections between components
chat_pipeline = Pipeline()
chat_pipeline.add_component("retriever", retriever)
chat_pipeline.add_component("prompt_builder", prompt_builder)
chat_pipeline.add_component("generator", llm)

chat_pipeline.connect("retriever.documents", "prompt_builder.documents")
chat_pipeline.connect("prompt_builder.prompt", "generator.prompt")

chat_history: List[ChatMessage] = []

query = "Where does Mark live?"
result = chat_pipeline.run({"retriever": {"query": query}, "prompt_builder": {"question": query, "chat_history": chat_history[-10:]}})
llm_response = result["generator"]["replies"][0]
print(f"llm_response: {llm_response}")
chat_history.append(ChatMessage.from_user(query))
chat_history.append(ChatMessage.from_assistant(llm_response))

query = "What did I ask you earlier and what was your response?"
result = chat_pipeline.run({"retriever": {"query": query}, "prompt_builder": {"question": query, "chat_history": chat_history[-10:]}})
llm_response = result["generator"]["replies"][0]
print(f"llm_response: {llm_response}")

import gradio as gr
chat_history: List[ChatMessage] = []
def chat(message, history):

    result = chat_pipeline.run(
        {"retriever": {"query": message}, "prompt_builder": {"question": message, "chat_history": chat_history[-10:]}})
    llm_response = result["generator"]["replies"][0]
    print(f"llm_response: {llm_response}")
    chat_history.append(ChatMessage.from_user(message))
    chat_history.append(ChatMessage.from_assistant(llm_response))

    return llm_response

questions = [
    "Where does Mark live?",
    "What language does Mark speak?",
    "Where was Jean born?",
    "Where does Jean live?",
]

demo = gr.ChatInterface(
    fn=chat,
    examples=questions,
    title="Ask me about Mark or Jean!",
)
demo.launch(share=True)