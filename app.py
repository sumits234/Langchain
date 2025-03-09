import os
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import

app = Flask(__name__)
api = Api(app)

# Load embeddings & vector store using the updated package
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

class ChatBot(Resource):
    def post(self):
        data = request.get_json()
        query = data.get('query')
        
        # Retrieve relevant documents for the query
        docs = retriever.get_relevant_documents(query)
        response = docs[0].page_content if docs else "No relevant information found"
        
        return jsonify({
            "query": query,
            "response": response
        })

api.add_resource(ChatBot, '/chat')

if __name__ == '__main__':
    app.run(debug=True)
