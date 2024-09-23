
from src.components.data_converter import DataConverter
from src.components.data_ingestion import VectorStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma 
from src.components.retrieval_generation import Generation
from flask import Flask, render_template, request, redirect, session
from werkzeug.utils import secure_filename
import os



# Load the embedding model 
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)

def vstore(file_path):
    pdf_processor = DataConverter(file_path)

    pdf_processor.convert_pdf_to_image()

    pdf_processor.convert_image_to_text()

    docs = pdf_processor.convert_text_to_doc()

    

    store = VectorStore()

    vector_path= store.vector_db(data=docs, embeddings=embeddings)

    return vector_path
    
def response(query, vector_path, model):
    
    vstore= Chroma(
            persist_directory=vector_path,
            embedding_function=embeddings
        )
    
    # Initialize the chat processor
    medical_chat = Generation(
        model_name=model,   # llama3-70b-8192 
        retriever= vstore.as_retriever())

    # Process a question in a specific session
    result= medical_chat.generate(
        input= query,
        session_id="Dhruv",  # Unique session ID
    )

    return result


app = Flask(__name__)

app.secret_key = 'super secret key'


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/process', methods=['POST'])
def process_document():
    
    
    os.makedirs('upload_doc', exist_ok= True)
    pdf_docs = request.files['pdf_docs']
    if pdf_docs:
        pdf_path = os.path.join("upload_doc/" + secure_filename(pdf_docs.filename))
        pdf_docs.save(pdf_path)
        vector_path = vstore(pdf_path)
        session['vector_path'] = vector_path
    return redirect('/chat')



@app.route('/chat', methods=['GET', 'POST'])
def chat():
    
    chat_history = []
    result = None

    vector_path = session.get('vector_path')
    if request.method == "POST":
        query = request.form['user_question']
        model = request.form.get('models')
        result= response(query, vector_path, model)
        chat_history.append(result)
        print(chat_history)
    return render_template('chat.html', chat_history= chat_history)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)  
