import streamlit as st
import pandas as pd

st.set_page_config(page_title="Hedge Fund Talent Map", layout="wide")

st.title("📊 Hedge Fund Talent Map")
st.caption("Explore fund manager movements and talent flows")

# File upload
uploaded_file = st.file_uploader("Upload hedge_fund_talent_map_final.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Filter columns
    with st.sidebar:
        st.subheader("🔍 Filter Options")

        selected_names = st.multiselect("Name", sorted(df["Name"].dropna().unique()))
        selected_firms = st.multiselect("Firm", sorted(df["Firm"].dropna().unique()))
        selected_titles = st.multiselect("Title", sorted(df["Title"].dropna().unique()))
        selected_strats = st.multiselect("Strategy", sorted(df["Strategy"].dropna().unique()))
        selected_moves = st.multiselect("Movement Type", sorted(df["Movement Type"].dropna().unique()))
        selected_locs = st.multiselect("Location", sorted(df["Location"].dropna().unique()))

    # Apply filters
    filtered = df.copy()
    if selected_names:
        filtered = filtered[filtered["Name"].isin(selected_names)]
    if selected_firms:
        filtered = filtered[filtered["Firm"].isin(selected_firms)]
    if selected_titles:
        filtered = filtered[filtered["Title"].isin(selected_titles)]
    if selected_strats:
        filtered = filtered[filtered["Strategy"].isin(selected_strats)]
    if selected_moves:
        filtered = filtered[filtered["Movement Type"].isin(selected_moves)]
    if selected_locs:
        filtered = filtered[filtered["Location"].isin(selected_locs)]

    # Show data
    st.markdown(f"### Results ({len(filtered)} rows)")
    st.dataframe(filtered, use_container_width=True)

    # Download filtered version
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download filtered CSV", csv, "filtered_talent_map.csv", "text/csv")
else:
    st.info("Upload the cleaned CSV file to begin.")
