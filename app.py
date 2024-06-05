import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import os 
import streamlit as st
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from langchain_community.document_loaders import PyPDFLoader
from tempfile import NamedTemporaryFile
# with open('docs/api_token.json', 'r') as api_file:
#     api_token_file = json.load(api_file)


openai_token = st.sidebar.text_input("OpenAI API Token", "sk-")
if len(openai_token) == 0:
    st.warning("Please enter your OpenAI API Token")
    st.stop()


# OpenAI API Token
#Open_api_token = api_token_file['Open_api_token']
Open_api_token = openai_token

class OpenAI_RAG:
    """
    Eine Klasse die ein OpenAI-Modell initialisiert und eine Frage beantwortet.
    Input: 
        - Ist die Frage als Varaible query

    Output:
        - Die Antwort auf die Frage
    
    """

    def __init__(self, Open_api_token: str, uploaded_file: str):
        self.Open_api_token = Open_api_token
        self.uploaded_file = uploaded_file


    def text_splitter(self):
        """
        Initialisiert den Text-Splitter

        Input:
            - None

        Output:
            - text_splitter: Ein Objekt des Text-Splitters
        """
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50, 
            length_function=len,
            
        )

        return text_splitter

    def loader_for_chunks(self, text_splitter):
        """
        Initialisiert den Loader für die Chunks mit der externen Datenquelle

        Input:
            - text_splitter: Ein Objekt des Text-Splitters aus der function text_splitter()
            - filepath: Der Pfad zur externen Datenquelle (z.B. eine PDF-Datei)

        Output:
            - chunks: Die Chunks der externen Datenquelle
        """
        # Annahme: Die Methode initialize() gibt Autor, Titel und Abstract zurück
           
        if self.uploaded_file:
            # Erstelle eine temporäre Datei
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                # Schreibe den Inhalt des hochgeladenen PDFs in die temporäre Datei
                temp_file.write(self.uploaded_file.read())

            # Initialisiere den PyPDFLoader mit dem Dateipfad der temporären Datei
            loader = PyPDFLoader(temp_file.name)
            chunks = loader.load_and_split()

            # Lösche die temporäre Datei
            os.unlink(temp_file.name)
        else:
            chunks = []
        
        return chunks

    def embedding(self):
        """
        Gibt ein Model mit Sentence-Embeddings zurück
        Input: 
            - None

        Output: 
            - embedding_function
        """

        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


        return embedding_function

    def initialise_chroma(self, chunks, embedding_function):
        """
        
        Initialisiert die Chroma Datenbank
        Input:
            - chunks: Die Chunks der Exteren Datenquelle
            - embedding_function: Ein Objekt des Sentence-Embeddings

        Output:
            - db: Die Chroma Datenbank
        """

        db = Chroma.from_documents(chunks, embedding_function)
        return db
    
    def retriever(self, db, query):
        """
        Initialisiert den Retriever für die externe Datenquellen und gibt die relevanten Dokumente zurück aus der Quelle
        filepath = '/Users/riccardo/Desktop/Repositorys_Github/LLM/Docs/merged.pdf'

        Input:
            - db: Die Chroma Datenbank
            - query: Die Frage
        
        Output:
            - retriever: Die relevanten Dokumente
        
        """

        retriever = db.as_retriever(search_kwargs={"k": 2})
        retriever.invoke(query)
        
        return retriever

    def llm_model(self):
        """
        Initialisiert das OpenAI-Modell. Hier wird das OpenAI modell genutzt für das RAG Modell
        
        Input:
            - None
        
        Output:
            - das LLM Modell von OpenAI
        """
        
        llm = ChatOpenAI(
            openai_api_key= Open_api_token,
            model_name = "gpt-3.5-turbo",
            temperature = 0.0,
            max_tokens = 300
        )

        return llm
        
    def qa_with_sources(self, query):
        """
        Die Funktion die die Frage beantwortet und die Quellen zurückgibt
        Input:
            - query: Die die Frage beinhalet
        Output:
            - qa_with_sources: Die Antwort auf die Frage und die Quellen
        
        """

        llm = self.llm_model()
        text_splitter_instance = self.text_splitter()
        chunks = self.loader_for_chunks(text_splitter_instance)
        embedding_instance = self.embedding()
        retriever_instance = self.retriever(Chroma.from_documents(chunks, embedding_instance), query)
        qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever_instance)
        
        return qa_with_sources.invoke(query)
    
# Streamlit Main
st.sidebar.title("OpenAI RAG")

st.title("OpenAI RAG")
st.write ("""This is a simple implementation of OpenAI's 
          Retrieval Augmented Generation (RAG) model. 
          The model is trained on a combination of 
          supervised and reinforcement learning. 
          It is capable of generating long-form answers 
          to questions, and can be used for a variety 
          of tasks, such as question answering, 
          summarization, and translation.""")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Hole den OpenAI-Token aus den Umgebungsvariablen
OPENAI_TOKEN = os.environ.get('OPENAI_TOKEN')

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

openai_rag = OpenAI_RAG(OPENAI_TOKEN, uploaded_file)

# Chat
if uploaded_file:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about: " + uploaded_file.name if uploaded_file else ""):
        if uploaded_file:
        # Append the user message to the history
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            with st.spinner("Thinking..."):  # Display spinner while processing
                antwort = openai_rag.qa_with_sources(prompt)
            
            with st.chat_message("assistant"):
                st.write(antwort["answer"])
                # append the assistant's response to the history
                st.session_state.messages.append({"role": "assistant", "content": antwort["answer"]})
        
            
    else:
        st.divider()
        with st.chat_message("assistant"):
            st.markdown("""
                        Welcome to Prime! 🤖 I am your personal document detective! Send me your PDFs and I will put them through their paces. From 
                        - "What's this about?" to 
                        - "What are the key points?" and even 
                        - "What's the scoop on topic X?" - 

                        I'm your man! 🕵️‍♂️ Uh, your bot. Never mind, you know. Let us crack your PDFs! 💼
                        Don't forget to upload a PDF! 📎
                        """)
else:   
    with st.chat_message("assistant"):
            st.markdown("""
                        Welcome to Prime! 🤖 I am your personal document detective! Send me your PDFs and I will put them through their paces. From 
                        - "What's this about?" to 
                        - "What are the key points?" and even 
                        - "What's the scoop on topic X?" - 

                        I'm your man! 🕵️‍♂️ Uh, your bot. Never mind, you know. Let us crack your PDFs! 💼
                        Don't forget to upload a PDF! 📎
                        """)