from langchain_groq import ChatGroq
from langchain_core.runnables import  RunnableWithMessageHistory
from langchain_chroma import Chroma 
from src.constant import GENERAL_PRACTITIONER
from src.constant import contextualize_q_system_prompt
from src.components.data_ingestion import VectorStore
from src.components.data_converter import DataConverter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from src.exception import GeneralPractitionerException
from src.logger import logging
from dotenv import load_dotenv
import os
import sys

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


class Generation:

    def __init__(self,model_name, retriever):
        try:

            self.llm = ChatGroq(
                model=model_name,
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            self.retriever = retriever
            history_aware_retriever = create_history_aware_retriever(self.llm, retriever, contextualize_q_prompt)

            qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", GENERAL_PRACTITIONER),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

            self.chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            self.store = {}

            self.system_message = GENERAL_PRACTITIONER
            # Define prompt template with placeholders
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_message),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ]
            )
        except Exception as e:
            raise GeneralPractitionerException(e, sys)

    def get_session_history(self, session_id: str)-> BaseChatMessageHistory:
        try: 
            if session_id not in self.store:
                self.store[session_id]= ChatMessageHistory()
            return self.store[session_id]
        except Exception as e:
            raise GeneralPractitionerException(e, sys)


    def generate(self, input: str,  session_id: str,):
        try:
            # Set up chain with message history management
            chain_with_message_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",    
            )

            # Invoke the chain with the provided question and session
            response = chain_with_message_history.invoke(
                {"input": input},
                config={"configurable": {"session_id": session_id}},
            )["answer"]
            
            return response

        except Exception as e:
            raise GeneralPractitionerException(e, sys)

if __name__ == "__main__":

    pdf_processor = DataConverter(file_path="uploads\\7210843.pdf")

    pdf_processor.convert_pdf_to_image()

    pdf_processor.convert_image_to_text()

    docs = pdf_processor.convert_text_to_doc()

    # Load the embedding model 
    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs)

    store = VectorStore()

    vector_path= store.vector_db(data=docs, embeddings=embeddings)
    # print(vector_path)
    vstore= Chroma(
            persist_directory=vector_path,
            embedding_function=embeddings
        )

    # Initialize the chat processor
    medical_chat = Generation(
        model_name="llama3-70b-8192",
        retriever= vstore.as_retriever())

    # Process a question in a specific session
    response = medical_chat.generate(
        input="About my Cholestrol levels, do i have to do anything? any exercise or diet?",
        session_id="Dhruv",  # Unique session ID
    )

    print(response)
