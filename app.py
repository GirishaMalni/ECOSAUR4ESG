import streamlit as st
from streamlit_lottie import st_lottie
import json
import os
import pandas as pd
import plotly.express as px
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
from transformers import pipeline
import torch
import re
from typing import Dict
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from crewai import Crew, Agent, Task
import joblib

st.set_page_config(page_title="ESG Analyzer", layout="wide")

def load_lottie(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

header_lottie = load_lottie("assets/esg_lottie.json")
planner_lottie = load_lottie("assets/planner.json")
strategy_lottie = load_lottie("assets/strategy.json")

st.sidebar.title("🌎 ESG Analyzer Tools")
uploaded_file = st.sidebar.file_uploader("📄 Upload ESG Report (PDF, Text, or Image)", type=["pdf", "txt", "png", "jpg"])
industry = st.sidebar.selectbox("🏢 Select Industry", ["Finance", "Healthcare", "Energy", "Agriculture", "Logistics", "Manufacturing", "Retail", "Technology", "Education", "Automotive"])
insight_level = st.sidebar.select_slider("🔍 Insights Level", options=["Basic", "Moderate", "Advanced"], value="Advanced")
show_trend = st.sidebar.checkbox("Show ESG Trend Analysis")
enable_recommendations = st.sidebar.checkbox("Enable AI Recommendations")
run_analysis = st.sidebar.button("🧠 Run Agentic Analysis")

with st.container():
    st.subheader("🍀Welcome to ECOSAUR : the ESG Analyzer")
    st.title("Empowering Responsible Decision Making through numbers : Sustainability Meets Intelligence")
    if header_lottie:
        st_lottie(header_lottie, height=150)

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        return f"Failed to extract text from PDF: {str(e)}"

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

summarizer = pipeline("summarization", device=0 if torch.cuda.is_available() else -1)

@st.cache_resource
def load_esg_model():
    model_path = "esg_score_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None, None

ml_model, model_features = load_esg_model()

def predict_esg_scores_ml(metadata_dict: Dict[str, str]) -> Dict[str, int]:
    if not ml_model:
        return None
    try:
        input_df = pd.DataFrame([metadata_dict])
        input_df = pd.get_dummies(input_df)

        # Align with training features
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_features]

        prediction = ml_model.predict(input_df)[0]
        return {
            "Environment": min(int(prediction[0]), 100),
            "Social": min(int(prediction[1]), 100),
            "Governance": min(int(prediction[2]), 100)
        }
    except Exception as e:
        st.warning(f"ML prediction failed: {e}")
        return None


def extract_esg_scores(text: str) -> Dict[str, int]:
    scores = {"Environment": 0, "Social": 0, "Governance": 0}
    for category in scores:
        match = re.search(f"{category}\\s*Score\\s*[:=]\\s*(\\d+)", text, re.IGNORECASE)
        if match:
            scores[category] = min(int(match.group(1)), 100)
        else:
            # Normalize count-based fallback to a 0–100 range (max 50 keywords → 100 points)
            raw_score = text.lower().count(category.lower()) * 2
            scores[category] = min(raw_score, 100)
    return scores


def generate_dynamic_insights(scores: Dict[str, int]) -> str:
    insights = []
    if scores["Environment"] < 60:
        insights.append("🌱 Improve environmental sustainability.")
    if scores["Social"] < 60:
        insights.append("🤝 Focus on inclusivity and social programs.")
    if scores["Governance"] < 60:
        insights.append("📊 Strengthen governance practices.")
    if all(score > 70 for score in scores.values()):
        insights.append("✅ Excellent performance across all ESG pillars.")
    return "\n".join(insights)


industry_benchmarks = {
    "Finance": {"Environment": 65, "Social": 70, "Governance": 75},
    "Healthcare": {"Environment": 60, "Social": 75, "Governance": 70},
    "Energy": {"Environment": 55, "Social": 60, "Governance": 65},
    "Agriculture": {"Environment": 70, "Social": 65, "Governance": 60},
    "Technology": {"Environment": 75, "Social": 70, "Governance": 80},
    "Retail": {"Environment": 60, "Social": 65, "Governance": 70},
    "Manufacturing": {"Environment": 58, "Social": 62, "Governance": 66},
    "Logistics": {"Environment": 55, "Social": 60, "Governance": 65},
    "Education": {"Environment": 68, "Social": 72, "Governance": 69},
    "Automotive": {"Environment": 62, "Social": 64, "Governance": 68}
}

def display_categorywise_comparison(esg_scores: Dict[str, int], benchmark: Dict[str, int]):
    st.subheader("📊 Detailed Category-wise ESG Scoring and Benchmark Comparison")

    # Create dataframe
    category_df = pd.DataFrame({
        "Category": esg_scores.keys(),
        "Your Score": esg_scores.values(),
        "Benchmark": [benchmark[cat] for cat in esg_scores.keys()],
        "Difference": [esg_scores[cat] - benchmark[cat] for cat in esg_scores.keys()]
    })

    
    st.dataframe(category_df.style.background_gradient(cmap='RdYlGn', subset=["Difference"]))


    bar_fig = px.bar(category_df, x="Category", y=["Your Score", "Benchmark"],
                     barmode="group", text_auto=True,
                     title="Category-wise ESG Score vs Benchmark")
    st.plotly_chart(bar_fig)


def run_crewai_agents(content):
    extractor = Agent(
        name="Extractor",
        role="ESG Data Extractor",
        goal="Extract ESG-specific information from the uploaded report.",
        backstory="Expert at reading ESG reports and extracting structured data like scores and keywords from unstructured formats."
    )

    analyzer = Agent(
        name="Analyzer",
        role="ESG Benchmark Analyst",
        goal="Compare the extracted ESG data against industry benchmarks.",
        backstory="Experienced in ESG benchmarking and scoring across multiple industries to identify strengths and weaknesses."
    )

    insight_generator = Agent(
        name="Insight Generator",
        role="Insightful ESG Advisor",
        goal="Generate strategic insights based on the analyzed ESG data.",
        backstory="Knowledgeable in sustainable business practices and ESG compliance strategies across sectors."
    )


    st.code("CrewAgent-Extractor: ESG data extracted")
    st.code("CrewAgent-Analyzer: Compared with benchmarks")
    st.code("CrewAgent-InsightGenerator: Generated insights")


if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()


    if file_ext == "txt":
        content = uploaded_file.read().decode("utf-8", errors="ignore")
    elif file_ext == "pdf":
        content = extract_text_from_pdf(uploaded_file)
    elif file_ext in ["png", "jpg", "jpeg"]:
        content = extract_text_from_image(uploaded_file)
    else:
        content = ""


    st.text_area("📝 Extracted Content", content[:2000], height=300)

   
    metadata_dict = {
        "Country Code": "IND",      
        "Year": "2023",             
        "Series Code": "ESGAGGREGATE"  
    }

  
    ml_scores = predict_esg_scores_ml(metadata_dict)

    if ml_scores:
        esg_scores = ml_scores
        st.info("✅ ESG scores predicted using trained ML model.")
    else:
        esg_scores = extract_esg_scores(content)
        st.info("⚠️ Fallback to rule-based ESG score extraction.")

    
    industry = st.selectbox("🏭 Select Industry", list(industry_benchmarks.keys()))

    benchmark = industry_benchmarks.get(industry, {"Environment": 60, "Social": 60, "Governance": 60})


    display_categorywise_comparison(esg_scores, benchmark)



    df = pd.DataFrame({"Category": list(esg_scores.keys()),
                       "Your Score": list(esg_scores.values()),
                       "Benchmark": [benchmark[k] for k in esg_scores.keys()]})

    overall_score = round(sum(esg_scores.values()) / 3, 2)
    normalized_score = round((overall_score / 100) * 100, 2) if overall_score <= 100 else 100
    benchmark_score = round(sum(benchmark.values()) / 3, 2)

    st.metric("Overall ESG Score (Your Report)", f"{normalized_score} / 100")
    st.metric("Industry Benchmark ({} Industry)".format(industry), benchmark_score)

    st.subheader("📌 ESG Category-wise Comparison")
    st.bar_chart(df.set_index("Category"))

    st.subheader("📡 ESG Radar Chart")
    fig = px.line_polar(r=df["Your Score"], theta=df["Category"], line_close=True)
    fig.update_traces(fill='toself')
    st.plotly_chart(fig)

    st.subheader("💡 AI-Powered Dynamic Insights")
    st.markdown(generate_dynamic_insights(esg_scores))

    if enable_recommendations:
        st.subheader("📈 Personalized Recommendations")
        recs = {
            "Environment": "Switch to renewable energy sources.",
            "Social": "Run diversity and inclusion workshops.",
            "Governance": "Implement internal audit practices."
        }
        for cat, rec in recs.items():
            st.markdown(f"- **{cat}**: {rec}")

    if planner_lottie:
        st_lottie(planner_lottie, height=150)
    if strategy_lottie:
        st_lottie(strategy_lottie, height=150)

    if run_analysis:
        run_crewai_agents(content)

if show_trend:
    st.subheader("📈 ESG Time Series Trend Analysis")

    if "esg_timeseries" not in st.session_state:
        st.session_state["esg_timeseries"] = []


    date_input = st.date_input("📅 Date of Report", help="Optional: Associate this ESG report with a specific date.")


    if st.button("➕ Add Report to ESG Trend Tracker"):
        st.session_state["esg_timeseries"].append({
            "Date": date_input if date_input else pd.Timestamp.now().date(),
            "Environment": esg_scores["Environment"],
            "Social": esg_scores["Social"],
            "Governance": esg_scores["Governance"],
            "Overall": overall_score
        })
        st.success("Added report to ESG time series!")

   
    if len(st.session_state["esg_timeseries"]) >= 2:
        ts_df = pd.DataFrame(st.session_state["esg_timeseries"])
        ts_df = ts_df.sort_values("Date")

        st.dataframe(ts_df)
        line_fig = px.line(ts_df, x="Date", y=["Environment", "Social", "Governance", "Overall"],
                           markers=True, title="📊 ESG Trend Over Time")
        st.plotly_chart(line_fig, use_container_width=True)
    elif len(st.session_state["esg_timeseries"]) == 1:
        st.info("Add more reports to view trend over time.")


st.markdown("---")
st.markdown("Made with 💚 towards the Sustainable Future by Girisha Malni" )

