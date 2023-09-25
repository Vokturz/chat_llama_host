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
from langchain.vectorstores import Weaviate
import weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.prompt import PromptTemplate

st.set_page_config(page_title="Chat with Documents", layout="wide")


_template = """[INST] <<SYS>> Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. <</SYS>>

Chat History:
{chat_history}
Follow Up Input: {question}
[/INST] Standalone question:"""

prompt_template = """[INST] <<SYS>> {system}
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context} <</SYS>>

Question: {question}
[/INST] Helpful Answer: """


col1, col2, col3 = st.columns(3)
system = col1.text_area("System", value="You are a helpful assistant expert in CHANGE_ME. Answer always in spanish.", height=160)
_template = col2.text_area("Condensed Question Template", value=_template, height=200)
prompt_template = col3.text_area("Prompt Template", value=prompt_template, height=200)

qa_prompt = PromptTemplate.from_template(prompt_template)
condense_question_prompt = PromptTemplate.from_template(_template)
st.title("Chat with Documents")


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files, host, index_name):
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = LocalAIEmbeddings(openai_api_key="NONE",
                                   openai_api_base=f"http://{host}:8090/v1")


    
    # Add documents
    vectordb = Weaviate.from_documents(splits, embeddings, weaviate_url=f"http://{host}:8000",
                index_name=index_name, by_text=False)

    # Load all documents
    client = weaviate.Client(f"http://{host}:8000")
    vectordb = Weaviate(client=client, index_name=index_name, text_key="text",
                         embedding=embeddings, by_text=False, attributes=["source"])

    # Define retriever
    retriever = vectordb.as_retriever(kwargs={'k': 4})
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

host = st.sidebar.text_input("Host", value="3.138.172.55")
if not host:
    st.info("Please add the Host address to continue.")
    st.stop()

index_name = st.sidebar.text_input("Index name")
if not index_name:
    st.info("Please add an index name to continue.")
    st.stop()

# Check if there are files
client = weaviate.Client(f"http://{host}:8000")

# index_name must be capitalized
index_name = index_name.capitalize()
index_exists = client.schema.exists(index_name)
files = []
if index_exists:
    st.sidebar.write(f"Index _{index_name}_ contains the following files.")
    for obj in client.data_object.get(class_name = index_name)['objects']:
        file = obj['properties']['source'].split('/')[-1]
        if file not in files:
            files.append(file)
            st.sidebar.markdown(f"- _{file}_")


uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)

if not uploaded_files and not index_exists:
    st.info("Please upload PDF documents to continue.")
    st.stop()

# for file in uploaded_files:
#     if file.name in files:
#         st.error(f"File {file.name} already exists.")
#         st.stop()


retriever = configure_retriever(uploaded_files, host, index_name)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = VLLMOpenAI(
    model_name="TheBloke/Llama-2-13B-chat-AWQ",
    openai_api_key="NONE",
    openai_api_base=f"http://{host}/v1",
    temperature=0,
    streaming=True,
    max_tokens=512,
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True,
    condense_question_prompt = condense_question_prompt,
    combine_docs_chain_kwargs={'prompt': qa_prompt.partial(system=system)}
)


if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

if st.sidebar.button("⚠️ **Delete Index**"):
    client.schema.delete_class(index_name)
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.rerun()

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
