import os
import tempfile
import streamlit as st
from langchain.llms import VLLMOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import LocalAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.prompt import PromptTemplate

_template = """[INST] <<SYS>> Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. <</SYS>>

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question: [/INST]"""
condense_question_prompt = PromptTemplate.from_template(_template)

prompt_template = """[INST] <<SYS>> {system}
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer: [/INST]"""
qa_prompt = PromptTemplate.from_template(prompt_template)

st.set_page_config(page_title="Chat with Documents")
st.title("Chat with Documents")

@st.cache_resource
def get_vectordb(host):
    embeddings = LocalAIEmbeddings(openai_api_key="NONE",
                                   openai_api_base=f"http://{host}:8090/v1")
    
    chroma_settings = Settings(chroma_api_impl="chromadb.api.fastapi.FastAPI",
                               chroma_server_host=host,
                               chroma_server_http_port="8000",
                               anonymized_telemetry=False)
    return Chroma(collection_name="agent_example",
                  client_settings=chroma_settings,
                  embedding_function=embeddings)

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files, host):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    vectordb = get_vectordb(host)
    vectordb.add_documents(splits, collection_name="agent_example")

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

host = st.sidebar.text_input("Host")
if not host:
    st.info("Please add the Host address to continue.")
    st.stop()

system = st.sidebar.text_area("System", value="You are a helpful assistant. Answer in no more than 40 words and in english.")

uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files, host)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = VLLMOpenAI(
    model_name="TheBloke/Llama-2-13B-chat-AWQ",
    openai_api_key="NONE",
    openai_api_base=f"http://{host}/v1",
    temperature=0,
    streaming=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=False,
    condense_question_prompt = condense_question_prompt,
    combine_docs_chain_kwargs={'prompt': qa_prompt.partial(system=system)}
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
