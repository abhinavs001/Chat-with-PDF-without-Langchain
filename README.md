# Chat-with-PDF-without-Langchain
This is a powerful PDF Document-based Question Answering System using using Retrieval Augmented Generation. By the power of LlamaIndex, Llama2 model using Gradient's LLM, seamlessly merged with superfast Apache Cassandra as a vector database.

<p><a href="https://colab.research.google.com/drive/16NTSHNU-kibiAp_CK876ddSGWpREjDpD?usp=drive_link" target="_blank"><img height="20" alt="Colab" src = "https://colab.research.google.com/assets/colab-badge.svg"></a></p>

### A Simple User interface is made with Streamlit
Deploy the file after setting up the virtual environment with:
```
streamlit run chatwithpdf.py
```

### Another file which uses FastAPI Integration
The system can be deployed and utilized through FastAPI which is the main.py file\n
First Install uvicorn
```
pip install "uvicorn[standard]"
```
Run the Server Program
```
uvicorn.run(app, host="127.0.0.1", port=5000)
```
