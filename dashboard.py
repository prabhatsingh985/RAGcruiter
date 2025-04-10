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

    # 🔍 Sidebar Filters
    st.markdown("#### 🔍 Filter Candidates")

    colA,colB=st.columns(2)
    

    with colA:        
        selected_candidate = st.selectbox("Select a Candidate (optional)", ["-- Show All --"] + df["Candidate Name"].unique().tolist())
        min_experience = st.slider("Minimum Experience (Years)", min_value=0, max_value=30, value=0)
    with colB:
        selected_tools = st.multiselect("Filter by Tools", sorted(set(
            tool.strip() for tools in df["Tools"].dropna() for tool in tools.split(",")
        )))
        min_match = st.slider("Minimum Match %", min_value=0, max_value=100, value=0, step=5)

    st.markdown("-------")    

    # Apply Filters
    if selected_candidate == "-- Show All --":
        
        
        df = df[df["Parsed Match %"] >= min_match]

        def extract_years(exp):
            return exp.count(":")


        df["Experience Years"] = df["Experience"].apply(extract_years)
        df = df[df["Experience Years"] >= min_experience]

        if selected_tools:
            df = df[df["Tools"].apply(lambda x: any(tool in x for tool in selected_tools) if isinstance(x, str) else False)]

        total = len(df)
        above_70 = df[df["Parsed Match %"] >= 70].shape[0]
        avg_match = round(df["Parsed Match %"].mean(), 2) if not df.empty else 0

        #with st.container():
            #st.markdown("## 📈 Summary Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Applications", total)
        col2.metric("Matches > 70%", above_70)
        col3.metric("Avg. Match %", f"{avg_match}%")

        #st.markdown("---")           

        st.markdown("#### 🏆 Top Candidates")
        topn = df.sort_values(by="Parsed Match %", ascending=False).head(5).reset_index()
        st.table(topn[["Candidate Name", "Filename", "Job Match Percentage"]])

        #st.markdown("## 📊 Visual Insights")
        col1, col2 = st.columns(2)

        with col1:
            #st.markdown("#### 🔧 Top Tools")
            tool_list = df["Tools"].dropna().str.split(",").explode().str.strip()
            tool_count = tool_list.value_counts().head(10).reset_index()
            tool_count.columns = ["Tool", "Count"]
            fig_tool = px.bar(tool_count, x="Tool", y="Count", title="Top Tools")
            st.plotly_chart(fig_tool, use_container_width=True)

            #st.markdown("#### ❤️ Interests Distribution")
            interest_list = df["Interests"].dropna().explode().str.split(",").explode().str.strip()
            interest_count = interest_list.value_counts().head(10).reset_index()
            interest_count.columns = ["Interest", "Count"]
            fig = px.pie(interest_count, names="Interest", values="Count", title="Top Interests")
            st.plotly_chart(fig, use_container_width=True)



        with col2:
            st.markdown("###### 🧠 Skill Match Count")
            st.bar_chart(df.set_index("Candidate Name")["Skill Match Count"])


            #st.markdown("#### 📄 Word Count Buckets")
            word_bins = pd.cut(df["Word Count"], bins=[0, 300, 500, 1000], labels=["<300", "300-500", ">500"])
            word_counts = word_bins.value_counts().reset_index()
            word_counts.columns = ["Word Range", "Count"]
            fig_pie = px.pie(word_counts, names="Word Range", values="Count", hole=0.4, title="Resume Length Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
    
        st.subheader("📋 Full Resume Overview")
        st.dataframe(df.drop(columns=["Parsed Match %"]), use_container_width=True)


        st.markdown("---")

    if selected_candidate != "-- Show All --" and not df.empty:
        df = df[df["Candidate Name"] == selected_candidate]
        st.markdown("## 👤 Candidate Profile")
        row = df.iloc[0]
        with st.container():
            col1, col2 = st.columns([3, 3])

            with col1:
                st.markdown(f"### {row['Candidate Name']}")
                st.markdown(f"**Match %:** {row['Parsed Match %']}%")
                st.markdown(f"**Word Count:** {row['Word Count']}")

                st.markdown("#### 💼 Experience")
                exp = row.get("Experience", {})
                result_dict={}
                for line in exp.strip().split('\n'):
                    match = re.match(r'(\d+):\s*(.*)', line)
                    if match:
                        key = match.group(1)
                        value = match.group(2).strip()
                        result_dict[key] = value
                        st.write(f"📅 {value}")

            with col2:
                st.markdown("#### 🔧 Skills")
                skills = row.get("Detected Skills", [])
                
                if isinstance(skills, list):
                    for skill in skills:
                        st.write(f"✅ {skill}")

                st.markdown("#### 🛠 Tools")
                tools = row.get("Tools", "")
                if tools:
                    items=[tool.strip() for tool in tools.split(",")]

                    cola,colb=st.columns(2)

                    for i,item in enumerate(items):
                        if i%2==0:
                            with cola:
                                st.write(f"🧰 {item}")
                        else:
                            with colb:
                                st.write(f"🧰 {item}")

                    #st.write(f"🧰 {items}")            

                st.markdown("#### Interests")
                interests = row.get("Interests", []).split(",")
                if isinstance(interests, list):
                    cola,colb=st.columns(2)
                    for i,interest in enumerate(interests):
                        if i%2==0:
                            with cola:
                                st.write(f"⭐ {interest}")
                        else:
                            with colb:
                                st.write(f"⭐ {interest}")
        


        st.markdown("---")

  
