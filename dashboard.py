# dashboard.py
import streamlit as st
import pandas as pd
import re
import plotly.express as px

def render_dashboard(df: pd.DataFrame):
    if df.empty:
        st.info("Upload resumes to view dashboard analysis.")
        return

    df["Parsed Match %"] = df["Job Match Percentage"].apply(
        lambda x: int(re.search(r'\d+', x).group(0)) if isinstance(x, str) and re.search(r'\d+', x) else 0
    )

    total = len(df)
    above_70 = df[df["Parsed Match %"] >= 70].shape[0]
    avg_match = round(df["Parsed Match %"].mean(), 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Applications", total)
    col2.metric("Applications > 70% Match", above_70)
    col3.metric("Average Match %", f"{avg_match}%")

    #st.markdown("---")

    st.subheader("🏆 Top 3 Candidates by Match %")
    top3 = df.sort_values(by="Parsed Match %", ascending=False).head(3)
    st.table(top3[["Candidate Name", "Filename", "Job Match Percentage"]])

    st.markdown("---")

    st.subheader("📊 Skill Match Count Distribution")
    st.bar_chart(df.set_index("Filename")["Skill Match Count"])

    st.subheader("📊 Match Percentage Distribution")
    st.bar_chart(df.set_index("Filename")["Parsed Match %"])

    # Word Count Distribution
    st.subheader("📄 Word Count Buckets")
    word_bins = pd.cut(df["Word Count"], bins=[0, 300, 500, 1000],
                       labels=["<300", "300-500", ">500"])
    word_counts = word_bins.value_counts().reset_index()
    word_counts.columns = ["Word Range", "Count"]
    fig_pie = px.pie(word_counts, names="Word Range", values="Count", hole=0.4,
                     title="Resume Length Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)


    st.subheader("📋 Full Resume Overview")
    st.dataframe(df.drop(columns=["Parsed Match %"]), use_container_width=True)
