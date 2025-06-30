
import streamlit as st
import pandas as pd
import re
from io import StringIO
import spacy
from transformers import pipeline
import torch
import subprocess

# ✅ Ensure model is available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


@st.cache_resource
def load_spacy_ner():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_bert_ner():
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )

def extract_entities(text, method="spaCy"):
    name, firm, strategy, movement, location, title = "", "", "", "", "", ""
    lowered = text.lower()

    strategy_keywords = ["macro", "long/short", "l/s equity", "credit", "quant", "multi-strategy",
                         "event-driven", "stat arb", "systematic", "equity", "fixed income",
                         "hedge", "volatility", "quant macro", "low-net", "healthcare", "value"]
    movement_keywords = {"launch": "Launch", "joined": "Join", "joining": "Join", "joins": "Join",
                         "spun out": "Spinout", "hired": "Hired", "promoted": "Promotion",
                         "departed": "Departure", "left": "Departure", "raising": "Fundraising",
                         "ex-": "Departure", "appointed": "Promotion", "started": "Launch",
                         "debuts": "Launch", "moving to": "Join", "onboards": "Join"}
    location_keywords = ["London", "New York", "Singapore", "Hong Kong", "Paris", "Tokyo", "Zurich"]
    title_keywords = ["CIO", "CEO", "Founder", "Co-Founder", "Portfolio Manager", "PM", "Managing Partner",
                      "Analyst", "Co-CIO", "Head of", "Investment Officer", "Chief Investment Officer",
                      "Partner", "Trader", "Director", "VP", "Principal", "COO", "CTO"]

    if method == "spaCy":
        doc = models['spacy'](text)
        ents = [(ent.text, ent.label_) for ent in doc.ents]
    else:
        results = models['bert_ner'](text)
        ents = [(ent['word'], ent['entity_group']) for ent in results]

    for ent_text, label in ents:
        if label in ("PER", "PERSON") and not name:
            name = ent_text
        elif label in ("ORG",) and not firm:
            firm = ent_text
        elif label in ("LOC", "GPE") and not location:
            location = ent_text

    for s in strategy_keywords:
        if s in lowered:
            strategy = s.title()
            break
    for k, v in movement_keywords.items():
        if k in lowered:
            movement = v
            break
    for loc in location_keywords:
        if loc.lower() in lowered:
            location = location or loc
            break
    for t in title_keywords:
        if t.lower() in lowered:
            title = t
            break

    return pd.Series({
        "Name": name, "Firm": firm, "Title": title,
        "Strategy": strategy, "Movement Type": movement,
        "Location": location, "Source Text": text
    })

st.set_page_config(page_title="Hedge Fund Talent Map", layout="wide")
st.title("📊 Hedge Fund Talent Map Extractor")

method = st.radio("Choose NER method:", ["spaCy", "BERT"], horizontal=True)
uploaded_file = st.file_uploader("Upload .txt file", type="txt")

models = {
    "spacy": load_spacy_ner(),
    "bert_ner": load_bert_ner()
}

if uploaded_file:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    raw_text = stringio.read()
    entries = [e.strip() for e in re.split(r"\n\s*\n|\n•|\n-", raw_text)
               if len(e.strip()) > 40 and not e.lower().startswith("http")]

    df = pd.DataFrame([extract_entities(e, method) for e in entries])
    df = df[df["Name"].str.len() > 1].drop_duplicates()

    st.success(f"Extracted {len(df)} entries")
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, "talent_map_cleaned.csv", "text/csv")
else:
    st.info("Upload a raw .txt file with fund movement reports")
