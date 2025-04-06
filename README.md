# 🌱 ECOSAUR4ESG – Sustainability Meets Intelligence  
**MULTIMODAL RAG AUTOMATION FOR ESG REPORT EVALUATION**  
*A Responsible AI Model where Greener Data Meets Smarter Decisions!*

---

## 🚀 Key Features

- 🧠 **CrewAI Agentic Workflow**: Modular AI agents for extracting, analyzing, benchmarking, and generating ESG insights.
- 🖼️ **Multimodal Parsing**: Processes text, charts, tables, and images from PDF, TXT, PNG, JPG reports.
- 📊 **Category-wise ESG Scoring**: Computes scores for Environment, Social, and Governance out of 100.
- 🧩 **Unified Multimodal Embedding Space**: Embeds text, image, graph, and tabular data for intelligent querying.
- 🧠 **Mixture of Insightful Experts (MoIE)**: Expert ensemble reasoning per ESG domain with an adaptive gating mechanism.
- 🔎 **RAG with LangChain + FAISS**: ESG queries retrieve relevant chunks using vector similarity and context-aware analysis.
- 📉 **Benchmark Comparison**: Real-time visual and tabular comparisons with industry ESG benchmarks.
- 📈 **Interactive Visualizations**: Radar plots, bar charts, and ESG JSON reports via Streamlit UI.

---

## 🧠 Architecture Highlights

1. **Input Preprocessing**
   - PDFs parsed via PyMuPDF; fallback to OCR using Tesseract for scanned content.
   - Tables parsed using Camelot, converted to JSON.
   - Images extracted and encoded via CLIP; graphs parsed or OCR'd.
   - Text is cleaned and chunked for retrieval.

2. **Unified Embedding Space**
   - Text, image, table, and layout embeddings fused using HuggingFace models + CLIP.
   - Stored in FAISS vector database for efficient retrieval.

3. **Gating Network (MoIE)**
   - Lightweight Transformer/MLP assigns weights to ESG experts.
   - Decision based on content type, semantic similarity, and info density.

4. **Retrieval-Augmented Generation (RAG)**
   - ESG queries embedded and matched with multimodal DB.
   - Retrieved chunks processed by:
     - 📥 Extractor Agent
     - 📊 Analyzer Agent
     - 📌 Benchmark Agent
     - 💡 Insight Generator Agent

5. **Insights & Output**
   - AI-generated ESG insights with benchmarking (via ScrapeGraphAI).
   - Outputs category-wise ESG scores, visual comparisons, and action recommendations.
   - Final output: JSON report + interactive visual dashboard.

---

## 🧰 Extensive Tech Stack

### 🤖 AI Frameworks & Agent Architecture
- **CrewAI** – Agent-based modular architecture
- **LangChain** – Orchestrating LLM chains and RAG pipelines
- **Hugging Face Transformers** – LLMs for QA, reasoning, summarization
- **MoIE (Mixture of Insightful Experts)** – Gated expert selection for E, S, G domains
- **ScrapeGraphAI** – External benchmarking agent using knowledge graphs

### 🧠 NLP & Language Models
- **Transformers (BERT, T5, GPT)** – For semantic understanding and insight generation
- **SentencePiece** – Subword tokenization
- **Torch** – For model loading and fine-tuning

### 📸 Computer Vision & Image Models
- **PyMuPDF (fitz)** – Image & text extraction from PDFs
- **pytesseract + PIL** – OCR for scanned PDFs and figures
- **CLIP** – Multimodal (image + text) embedding generation

### 🗂️ Data Extraction & Processing
- **PyPDF2** – PDF text parsing
- **Camelot** – Table extraction from PDFs
- **regex** – Pattern extraction
- **Pandas** – Dataframe-based manipulation
- **joblib** – Efficient model serialization

### 🔍 Vector Search & Embedding Space
- **FAISS (CPU)** – Fast vector similarity search
- **Unified Embedding Pipeline** – Custom logic for combining text, image, graph, and table embeddings

### 📈 Scoring & Machine Learning
- **scikit-learn** – Classification, score normalization
- **xgboost** – ESG score classification & regression

### 🖼️ Visualization
- **Streamlit** – Web interface for file uploads and visualization
- **Plotly** – Radar plots, bar charts, pie charts, score visuals

---

## 📦 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
🤝 Contributing
Pull requests, issues, and forks are welcome. Help us make ESG reporting smarter and more responsible.

📬 Contact
Made with ❤️ by Girisha Malni
🔗 GitHub: https://github.com/GirishaMalni/ECOSAUR4ESG
📧 Email: [Contact via GitHub Issues]

📄 License
This project is licensed under the MIT License – feel free to use, modify, and distribute.


