# ECOSAUR4ESG
MULTIMODAL RAG AUTOMATION FOR ESG REPORT EVALUATION
# 🌱 ECOSAUR4ESG – ESG Report Intelligence Engine



**ECOSAUR4ESG** is an intelligent multimodal ESG (Environmental, Social, and Governance) analyzer that extracts, evaluates, and visualizes ESG-related data from unstructured reports using AI-powered agents, unified multimodal embeddings, and insightful scoring techniques.

---

## 🚀 Features

- 🔍 Extract ESG data from PDF reports using **CrewAI agents**
- 🤖 Use **Mixture of Insightful Experts (MoIE)** to enhance analysis across E, S, G domains
- 🔗 Leverage **Unified Multimodal Embedding Space** for text & image alignment
- 📊 Compute overall & category-wise ESG scores out of 100
- 📈 Visual comparison with ESG benchmarks using colorful charts & tables
- 🎯 Final ESG output JSON + graphical insights

---

## 🧠 Architecture Highlights

- **Crew AI Agents**: Modular agent-based architecture to extract, clean, analyze, and report ESG data
- **MoIE**: Expert selectors to handle domain-specific insights
- **LLM + Vision + Embedding Fusion**: Unified space to process multimodal (text + visual) content
- **Streamlit**: Intuitive frontend for uploading and visualizing results

---

## 🛠️ Tech Stack

| Component               | Tool / Framework              |
|------------------------|-------------------------------|
| 🧠 AI Orchestration     | **CrewAI**, **MoIE**, **LangChain** |
| 📄 PDF Handling         | **PyMuPDF**, **pdf2image**     |
| 🤖 LLM & Vision         | **Ollama (Qwen-VL)**, **OpenAI**, **CLIP** |
| 📐 Embeddings           | **Unified Multimodal Embedding Space** |
| 📊 Visualization        | **Plotly**, **Matplotlib**, **Streamlit** |
| 🧮 Scoring Logic        | **Pydantic**, **NumPy**, **Custom Benchmarks** |
| 🔗 Version Control      | **Git**, **GitHub**             |

---

## 🖼 Sample Output

- ✅ ESG Score: `82.3 / 100`
- ✅ Category Scores:
  - Environment: `85 / 100`
  - Social: `78 / 100`
  - Governance: `84 / 100`
- 📊 Visualized graphs comparing each category with industry benchmarks
- 📄 Final JSON with extracted ESG data

---

## 📦 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
🤝 Contributing
Contributions are welcome! Please raise issues or submit a pull request.

📃 License
This project is licensed under the MIT License.

📬 Contact
Made with ❤️ by Girisha Malni
🔗 GitHub Repo: https://github.com/GirishaMalni/ECOSAUR4ESG

yaml
Copy
Edit

---

Let me know if you'd like a version with badges (build passing, license, etc.) or want a cleaner minimalist version too.
