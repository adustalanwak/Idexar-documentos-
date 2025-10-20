# 📚 PDF Indexer Pro - Sistema Inteligente de Consulta de PDFs

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)
![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)
![RAG](https://img.shields.io/badge/Architecture-RAG-purple.svg)

Una aplicación de escritorio inteligente que indexa y consulta documentos PDF usando modelos de lenguaje local (LLM) mediante Ollama y tecnología RAG (Retrieval-Augmented Generation).

## 🚀 Características Principales

- **📄 Indexación Inteligente**: Extrae texto de PDFs (digitales y escaneados)
- **🤖 Resumen Automático**: Genera resúmenes con modelos Llama3
- **🔍 Consulta Semántica**: Búsqueda conversacional en documentos indexados
- **💾 Almacenamiento Vectorial**: Usa FAISS para búsquedas eficientes
- **🖥️ Interfaz Gráfica**: Aplicación de escritorio con PyQt5
- **🌐 Soporte Español**: OCR y procesamiento optimizado para español

## 🛠️ Tecnologías Utilizadas

### **Backend**
- **Python 3.8+** - Lenguaje principal
- **PyQt5** - Interfaz gráfica de usuario
- **PyMuPDF (fitz)** - Procesamiento de PDFs
- **pytesseract** - OCR para documentos escaneados
- **Ollama** - Modelos de lenguaje local
- **LangChain** - Framework para aplicaciones con LLMs
- **FAISS** - Almacenamiento y búsqueda vectorial
- **SQLite** - Base de datos para metadatos

### **Modelos**
- **Llama3** - Modelo de lenguaje principal
- **OllamaEmbeddings** - Generación de embeddings

## ⚙️ Instalación

### Prerrequisitos
1. **Ollama instalado y ejecutándose**
```bash
# Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Descargar modelo Llama3
ollama pull llama3
