from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os
import sys
from src.components.data_converter import DataConverter
from src.entity.config_entity import DirPath
from src.exception import GeneralPractitionerException
from src.logger import logging
import warnings
warnings.filterwarnings("ignore")

class VectorStore:

    def __init__(self,):
        try:
            self.vectors_path = None
        except Exception as e:
            raise GeneralPractitionerException(e, sys)
        
    def vector_db(self, data, embeddings):
        try:
            logging.info("Start ingesting data into vector store")
            name = DirPath()
            base_directory = name.get_chroma_dir()
            vectors_path = os.path.join(base_directory)
            os.makedirs(base_directory, exist_ok=True)
            Chroma.from_documents(data, embeddings, persist_directory= vectors_path)
            logging.info("Ingesting data into vector store completed.")
        except Exception as e:
            raise GeneralPractitionerException(e, sys)
        return vectors_path
    
if __name__ == '__main__':

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
