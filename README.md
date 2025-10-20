https://img.shields.io/badge/Python-3.8+-blue.svg https://img.shields.io/badge/GUI-PyQt5-green.svg https://img.shields.io/badge/LLM-Ollama-orange.svg https://img.shields.io/badge/Architecture-RAG-purple.svg

Una aplicación de escritorio inteligente que indexa y consulta documentos PDF usando modelos de lenguaje local (LLM) mediante Ollama y tecnología RAG (Retrieval-Augmented Generation).

🚀 Características Principales
📄 Indexación Inteligente: Extrae texto de PDFs (digitales y escaneados)

🤖 Resumen Automático: Genera resúmenes con modelos Llama3

🔍 Consulta Semántica: Búsqueda conversacional en documentos indexados

💾 Almacenamiento Vectorial: Usa FAISS para búsquedas eficientes

🖥️ Interfaz Gráfica: Aplicación de escritorio con PyQt5

🌐 Soporte Español: OCR y procesamiento optimizado para español

🛠️ Tecnologías Utilizadas
Backend: Python 3.8+, PyQt5, PyMuPDF, pytesseract, Ollama, LangChain, FAISS, SQLite
Modelos: Llama3, OllamaEmbeddings

⚙️ Instalación Rápida
bash
# 1. Instalar Ollama y modelo
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3

# 2. Instalar Tesseract (OCR)
# Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-spa
# macOS: brew install tesseract

# 3. Instalar dependencias Python
pip install PyQt5 PyMuPDF pytesseract Pillow langchain langchain-community faiss-cpu requests
🎯 Uso
Ejecutar aplicación: python main.py

Indexar PDFs: Pestaña "Indexar PDFs" → Seleccionar archivo

Consultar: Pestaña "Consultar" → Escribir pregunta → Obtener respuesta

🏗️ Arquitectura
text
PDF → Extracción Texto/OCR → Chunking → Embeddings → FAISS Vector Store
                                                         ↓
Consulta Usuario → Búsqueda Semántica → Ollama LLM → Respuesta + Fuentes
📝 Configuración
Modelo: Llama3 (configurable)

Chunk size: 2000 caracteres

Búsqueda: Top 3 chunks relevantes

OCR: Soporte para español

🐛 Solución de Problemas
Ollama no responde: Ejecutar ollama serve
Tesseract no encontrado: Verificar instalación y PATH
Dependencias faltantes: pip install --upgrade -r requirements.txt
