import os
import time
import nest_asyncio
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
import cohere
from langchain import hub
from langchain_groq import ChatGroq
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = 'end-to-end-rag'

# SQLAlchemy base for document metadata
Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String, nullable=False)
    vectorstore_id = Column(String)


class RAG:
    def __init__(self):
        self.vectorstore_index_name = 'nvli-rag-project'  # Initializing vector store name

        # Embeddings Model
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
        )

        # LLM Model for output
        self.groq_llm = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model='llama3-8b-8192',
            temperature=0.2
        )

        # Document Splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        self.create_pinecone_index(self.vectorstore_index_name)  # Creating Pinecone vector database

        # Vector Store for storing embeddings
        self.vectorstore = PineconeVectorStore(
            index_name=self.vectorstore_index_name,
            embedding=self.embeddings,
            pinecone_api_key=os.getenv('PINECONE_API_KEY')
        )

        # RAG Prompt
        self.rag_prompt = hub.pull(
            "rlm/rag-prompt", 
            api_key=os.getenv("LANGSMITH_API_KEY")
        )

        # Guardrails using RailsConfig
        config = RailsConfig.from_path('/home/kushan/something_ai/config.yml')
        self.guardrails = RunnableRails(config=config, llm=self.groq_llm)
        
        # Initialise cohere client
        self.cohere_client = cohere.Client(api_key=os.getenv('COHERE_API_KEY'))

        # Initialize SQLAlchemy database
        self.init_db()


    def rerank_docs(self, query, docs):
        doc_text = [doc.page_content for doc in docs]

        response = self.cohere_client.rerank(
            query=query,
            documents=doc_text,
            top_n=5
        )

        ranked_docs = [docs[result['index']] for result in response['results']]
        return ranked_docs
    

    # Initialize the SQLAlchemy engine and session
    def init_db(self):
        self.engine = create_engine('sqlite:///docs_metadata.db')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()


    # Check if a document already exists in the database
    def doc_exists(self, file_path):
        return self.session.query(Document).filter_by(file_path=file_path).first() is not None
    

    # Add document metadata to the database after processing
    def add_doc_to_db(self, file_path, status, vectorstore_id=None):
        doc = Document(file_path=file_path, status=status, vectorstore_id=vectorstore_id)
        self.session.add(doc)
        self.session.commit()


    # Create the Pinecone index for the vector store
    def create_pinecone_index(self, vectorstore_index_name):
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        spec = ServerlessSpec(cloud='aws', region='us-east-1')

        if vectorstore_index_name not in pc.list_indexes().names():
            pc.create_index(
                vectorstore_index_name,
                dimension=1536,
                metric='cosine',
                spec=spec
            )

            while not pc.describe_index(vectorstore_index_name).status['ready']:
                time.sleep(1)


    # Load documents into the vector store and track them in the database
    def load_docs_into_vectorstore(self, file_path):
        if self.doc_exists(file_path):
            print(f"Document {file_path} is already loaded.")
            return

        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load_and_split()
                
            split_docs = self.text_splitter.split_documents(documents)

            # Add metadata to the documents before adding to the vector store
            for doc in split_docs:
                doc.metadata = {
                    'title': self.extract_title(doc.page_content),
                    'author': self.extract_author(doc.page_content),
                    'source': file_path
                }

            self.vectorstore.add_documents(split_docs)
            self.add_doc_to_db(file_path, 'loaded', vectorstore_id=self.vectorstore_index_name)
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            self.add_doc_to_db(file_path, 'failed')


    # Extract title from content
    def extract_title(self, content):
        # Placeholder logic for title extraction (can be refined)
        return "Sample Title"

    # Extract author from content
    def extract_author(self, content):
        # Placeholder logic for author extraction (can be refined)
        return "Sample Author"


    # Find top N similar books based on document embeddings
    def find_similar_books(self, doc_embedding, top_n=20):
        """Find top N similar books based on embeddings."""
        # try:
        similar_docs = self.vectorstore.similarity_search_by_vector_with_score(
            embedding=doc_embedding,
            k=top_n
        )
        try:
            similar_books = set()
            for doc, score in similar_docs:
                book_name = doc.metadata['source'].split('/')[-1]
                similar_books.add(book_name)
            return similar_books
        except:
            return 'No Similar Books Exist!'


    # Format documents for output
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    

    # Create a retrieval chain for answering questions using RAG
    def create_retrival_chain(self, query):
        self.retriver = self.vectorstore.as_retriever()  # Retrieve relevant docs
        retrived_docs = self.retriver.get_relevant_documents(query=query)

        self.rag_chain = (
            {
                'context': self.retriver | self.format_docs, 'question': RunnablePassthrough()
            }
            | self.rag_prompt
            | self.groq_llm
            | StrOutputParser()
        )
        self.rag_chain = self.guardrails | self.rag_chain 
    
    def create_query(self, query):
        self.retriver = self.vectorstore.as_retriever()  
        retrived_docs = self.retriver.get_relevant_documents(query=query)
        
        metadata_list = []

        for document in retrived_docs:
            metadata = document.metadata
            page_number = metadata['page']
            source_file = metadata['source']
            metadata_list.append((f"Page: {page_number}, Source: {source_file}"))
        return metadata_list

    # Function to handle queries using RAG and recommend similar books
    nest_asyncio.apply()
    def qa(self, query):
        # Step 1: Retrieve the metadata
        metadata = self.create_query(query)

        # Step 2: Modify query to include metadata for the final answer
        query = query + f'Extract book name from the {metadata} mention that as the book name in the final answer and if you are unable to identify page numbers on your own again refer to {metadata}. Make sure you do not forget to mention book name and page number. And write them at the end after the answer in clean format. Answer in 200 words, Write the book name and page number in italics '

        # Step 3: Retrieve the documents and create the retrieval chain
        self.create_retrival_chain(query)

        # Step 4: Get the final answer from the RAG system
        answer = self.rag_chain.invoke(query)

        # Step 5: Extract embedding of the retrieved documents' content for similarity search
        doc_embedding = self.embeddings.embed_query(answer)

        # Step 6: Find similar books based on the embedding of the retrieved documents
        similar_books = self.find_similar_books(doc_embedding)

        # Step 7: Format the final answer and recommendations
        # recommendations = "\n\nRecommended Similar Books:\n" + "\n".join(similar_books)
        return {
                "answer": answer,
                "similar_books" :similar_books}


if __name__ == "__main__":
    rag = RAG()
    for book in os.listdir('/home/kushan/something_ai/Test Books'):
        rag.load_docs_into_vectorstore('/home/kushan/something_ai/Test Books' + '/' + book)