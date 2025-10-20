import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, \
    QTextEdit, QListWidget, QFileDialog, QMessageBox, QProgressBar
from PyQt5.QtCore import Qt
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import sqlite3
import requests
import json
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configuración de Ollama
OLLAMA_URL =   # Ajusta IP si es necesario
MODEL = "llama3"
llm = Ollama(model=MODEL, base_url="http://localhost/api/generate")  # Para LangChain
embeddings = OllamaEmbeddings(model=MODEL, base_url="http://localhost/api/generate")

# Base de datos SQLite para metadatos
DB_FILE = "pdf_index.db"
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS pdfs (id INTEGER PRIMARY KEY, path TEXT, summary TEXT)''')
conn.commit()

# Vector store para embeddings (FAISS)
VECTOR_STORE_PATH = "faiss_index"
try:
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
except:
    vector_store = None  # Se creará al indexar el primero


def extract_text_from_pdf(pdf_path):
    """Extrae texto preciso, con OCR para escaneados, optimizado para español."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text.strip():  # Texto digital
            text += page_text + "\n"
        else:  # OCR para escaneados
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(img, lang='spa')  # Soporte para español
            text += ocr_text + "\n"
    doc.close()
    return text.strip()


def generate_summary(text):
    """Genera resumen optimizado con prompt claro."""
    prompt = (
            "Resumir de forma clara y concisa el siguiente contenido. Enfócate en puntos clave: "
            "tema principal, entidades involucradas, fechas importantes y conclusiones. "
            "Mantén el resumen en menos de 300 palabras: " + text[:5000]  # Límite para eficiencia
    )
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return json.loads(response.text)['response']
    return "Error al generar resumen."


def index_pdf(pdf_path, progress_bar):
    """Indexa PDF: extrae texto, resume, chunking y embeddings."""
    global vector_store
    text = extract_text_from_pdf(pdf_path)
    summary = generate_summary(text)

    # Almacena metadatos
    cursor.execute("INSERT INTO pdfs (path, summary) VALUES (?, ?)", (pdf_path, summary))
    conn.commit()

    # Chunking con LangChain
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in chunks]

    # Embeddings y vector store
    if vector_store is None:
        vector_store = FAISS.from_documents(documents, embeddings)
    else:
        vector_store.add_documents(documents)
    vector_store.save_local(VECTOR_STORE_PATH)

    progress_bar.setValue(100)  # Progreso completado
    return summary


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Indexador de PDFs con Ollama - Versión Pro")
        self.setGeometry(100, 100, 800, 600)

        # Tabs
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Tab 1: Indexar
        index_tab = QWidget()
        index_layout = QVBoxLayout()
        self.index_button = QPushButton("Seleccionar y Indexar PDF")
        self.index_button.clicked.connect(self.index_pdf_action)
        index_layout.addWidget(self.index_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        index_layout.addWidget(self.progress_bar)

        self.pdf_list = QListWidget()
        self.update_pdf_list()
        index_layout.addWidget(QLabel("PDFs Indexados:"))
        index_layout.addWidget(self.pdf_list)

        index_tab.setLayout(index_layout)
        tabs.addTab(index_tab, "Indexar PDFs")

        # Tab 2: Consultar
        query_tab = QWidget()
        query_layout = QVBoxLayout()
        query_layout.addWidget(QLabel("Pregunta sobre PDFs indexados:"))
        self.query_input = QLineEdit()
        query_layout.addWidget(self.query_input)

        self.query_button = QPushButton("Consultar")
        self.query_button.clicked.connect(self.query_action)
        query_layout.addWidget(self.query_button)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        query_layout.addWidget(QLabel("Resultados:"))
        query_layout.addWidget(self.result_display)

        query_tab.setLayout(query_layout)
        tabs.addTab(query_tab, "Consultar")

    def update_pdf_list(self):
        self.pdf_list.clear()
        cursor.execute("SELECT path, summary FROM pdfs")
        for path, summary in cursor.fetchall():
            self.pdf_list.addItem(f"{path} - Resumen: {summary[:50]}...")

    def index_pdf_action(self):
        pdf_path = QFileDialog.getOpenFileName(self, "Selecciona PDF", "", "PDF files (*.pdf)")[0]
        if pdf_path:
            self.progress_bar.setValue(50)  # Progreso intermedio
            try:
                summary = index_pdf(pdf_path, self.progress_bar)
                self.update_pdf_list()
                QMessageBox.information(self, "Éxito", f"PDF indexado. Resumen: {summary[:200]}...")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
            finally:
                self.progress_bar.setValue(0)

    def query_action(self):
        query = self.query_input.text()
        if query and vector_store:
            # Configura RAG con prompt optimizado
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="Basado en el siguiente contexto de PDFs: {context}\n\nResponde claramente a: {question}"
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Top 3 chunks relevantes
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template}
            )
            result = qa_chain({"query": query})
            response = result['result']
            sources = [doc.metadata['source'] for doc in result['source_documents']]

            self.result_display.append(f"Pregunta: {query}\nRespuesta: {response}\nFuentes: {', '.join(sources)}\n\n")
        else:
            QMessageBox.warning(self, "Error", "Indexa PDFs primero o ingresa una pregunta.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())