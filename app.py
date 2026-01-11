import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import re

# Add src to path if needed
sys.path.append(os.path.join(os.getcwd(), 'src'))
from config import *

# Page Config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("assets/style.css")

# Helper Functions
def highlight_text(text, query):
    if not query or not isinstance(text, str):
        return text
    # Case-insensitive highlight using regex
    pattern = re.compile(f"({re.escape(query)})", re.IGNORECASE)
    return pattern.sub(r"<mark style='background-color: #D4AF37; color: #0A1628; padding: 2px; border-radius: 2px; font-weight: bold;'>\1</mark>", text)

# Data Loading
@st.cache_data(ttl=60)  # Refresh every 60 seconds to pick up config changes
def load_data():
    file_path = "data/hansard_final_analyzed.csv"
    if not os.path.exists(file_path):
        st.error(f"Data file not found at {file_path}")
        return pd.DataFrame(), pd.DataFrame()
    
    df = pd.read_csv(file_path, parse_dates=["Date"])
    
    # Cleaning Topic Labels
    def clean_topic_name(row):
        try:
            topic_id = int(float(row['Topic']))  # Handle both string and float
        except (ValueError, TypeError):
            return "Unknown Topic"

        # 1. Check manual override in TOPIC_MAP
        if topic_id in TOPIC_MAP:
            return TOPIC_MAP[topic_id]
        
        # 2. Advanced cleaning for raw names
        raw_name = str(row['Topic_Label'])
        
        # Remove numbers like "14_", "Topic 14", or "-1_"
        clean_name = re.sub(r'^(-?\d+_|Topic\s+\d+)', '', raw_name).strip()
        
        # If the name is empty after stripping (e.g., was just "Topic 21")
        if not clean_name:
            return f"Topic {topic_id}"
            
        # Clean underscores and title case
        clean_name = clean_name.replace('_', ' ').strip().title()
        return clean_name

    df["Topic_Label_Friendly"] = df.apply(clean_topic_name, axis=1)
    
    # Apply Hard Filter for Noise Topics by default
    # Convert Topic to int for proper comparison with NOISE_TOPICS list
    df["Topic_Int"] = df["Topic"].apply(lambda x: int(float(x)) if pd.notna(x) else -999)
    df_clean = df[~df["Topic_Int"].isin(NOISE_TOPICS)].copy()
    
    return df, df_clean

# df_master contains everything, df_filtered contains only thematic topics
df_master, df_filtered = load_data()

# Header
st.markdown(f"""
    <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 30px; padding: 35px; background-color: {COLOR_NAVY}; border-radius: 20px; border: 1px solid {COLOR_GOLD}33; box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
        <div style='text-align: center;'>
            <h1 style='margin: 0; padding: 0; font-size: 4em; letter-spacing: 2px; font-family: "Playfair Display", serif; color: {COLOR_WHITE};'>
                Parl<span style='color: {COLOR_GOLD}; text-shadow: 0 0 15px {COLOR_GOLD}88; font-style: italic;'>AI</span>ment
            </h1>
            <p style='margin: 10px 0 0 0; color: {COLOR_GOLD}; font-size: 1.1em; letter-spacing: 3px; text-transform: uppercase; font-family: "Lato", sans-serif; opacity: 0.8;'>
                Intelligence • Accountability • Governance
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# High Level Metrics
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.metric("Thematic Speeches", f"{len(df_filtered):,}")
with m_col2:
    st.metric("Unique Speakers", len(df_filtered["Speaker"].unique()))
with m_col3:
    evasive_count = (df_filtered["Stance"] == "Evasive").sum()
    st.metric("Evasive Flagged", f"{evasive_count:,}", delta=f"{(evasive_count/len(df_filtered)*100):.1f}% Rate")
with m_col4:
    date_range = f"{df_filtered['Date'].min().year} - {df_filtered['Date'].max().year}"
    st.metric("Date Range", date_range)

st.divider()

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 The Pulse", "📊 The Stance", "🔍 The Evidence", "👤 MP Insights", "📜 Methodology"])

with tab1:
    st.header("Temporal Topic Analysis")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Controls")
        all_topic_options = sorted([str(x) for x in df_filtered["Topic_Label_Friendly"].dropna().unique()])
        selected_topics = st.multiselect(
            "Select Topics to Visualize",
            options=all_topic_options,
            default=all_topic_options[:5] if all_topic_options else []
        )
        
        show_evasiveness = st.checkbox("Overlay Evasiveness Index", value=False, 
                                       help="Show the percentage of speeches flagged as 'Evasive' over time.")
        
        time_res = st.radio("Time Resolution", ["Monthly", "Daily"], index=0)

    with col1:
        # Prepare data
        df_tab1 = df_filtered[df_filtered["Topic_Label_Friendly"].isin(selected_topics)].copy()
        
        if time_res == "Monthly":
            df_tab1["Time"] = df_tab1["Date"].dt.to_period("M").dt.to_timestamp()
        else:
            df_tab1["Time"] = df_tab1["Date"]
            
        # Topic counts
        df_counts = df_tab1.groupby(["Time", "Topic_Label_Friendly"]).size().reset_index(name="Count")
        
        if not df_counts.empty:
            fig = px.line(
                df_counts, 
                x="Time", 
                y="Count", 
                color="Topic_Label_Friendly",
                title="Topic Frequency Over Time",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            
            if show_evasiveness:
                # Calculate Evasiveness Rate
                df_evasive = df_tab1.groupby("Time").apply(
                    lambda x: (x["Stance"] == "Evasive").sum() / len(x) * 100 if len(x) > 0 else 0
                ).reset_index(name="Evasiveness_Rate")
                
                fig.add_trace(go.Scatter(
                    x=df_evasive["Time"], 
                    y=df_evasive["Evasiveness_Rate"],
                    name="Evasiveness Index (%)",
                    line=dict(color=COLOR_AMBER, width=4, dash='dot'),
                    yaxis="y2"
                ))
                
                fig.update_layout(
                    yaxis2=dict(
                        title="Evasiveness Index (%)",
                        overlaying="y",
                        side="right",
                        range=[0, 100]
                    )
                )

            fig.update_layout(
                hovermode="x unified",
                paper_bgcolor=COLOR_NAVY,
                plot_bgcolor=COLOR_NAVY,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data found for selected topics.")
        
        st.info("💡 **Insight:** Notice how certain topics spike during specific parliamentary sessions. A high 'Evasiveness Index' during these spikes often indicates controversial policy debates.")

with tab2:
    st.header("The Accountability Dashboard")
    st.markdown("---")
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.subheader("Stance Distribution by Party")
        
        # Aggregate data
        df_stance = df_filtered.groupby(["Party", "Stance"]).size().reset_index(name="Count")
        
        # Sort by total count
        party_order = df_filtered["Party"].value_counts().index.tolist()
        
        if not df_stance.empty:
            fig_bar = px.bar(
                df_stance, 
                x="Party", 
                y="Count", 
                color="Stance",
                color_discrete_map=STANCE_COLORS,
                title="Party Sentiment Profile",
                category_orders={"Party": party_order},
                template="plotly_dark",
                barmode="stack"
            )
            
            fig_bar.update_layout(
                paper_bgcolor=COLOR_NAVY,
                plot_bgcolor=COLOR_NAVY,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.subheader("The Evasiveness Heatmap")
        
        # Filter for top thematic topics to keep heatmap readable and stable
        # We pick top topics from the entire thematic dataset to ensure stability
        top_topics = df_filtered["Topic_Label_Friendly"].value_counts().nlargest(10).index.tolist()
        df_heat = df_filtered[df_filtered["Topic_Label_Friendly"].isin(top_topics)]
        
        if not df_heat.empty:
            # Calculate Evasiveness Index per Topic/Party
            heat_data = df_heat.groupby(["Topic_Label_Friendly", "Party"]).apply(
                lambda x: (x["Stance"] == "Evasive").sum() / len(x) * 100
            ).unstack(fill_value=0)
            
            fig_heat = px.imshow(
                heat_data,
                labels=dict(x="Party", y="Topic", color="Evasiveness %"),
                x=heat_data.columns,
                y=heat_data.index,
                color_continuous_scale="YlOrRd",
                title="Evasiveness Index by Topic & Party",
                template="plotly_dark",
                aspect="auto"
            )
            
            fig_heat.update_layout(
                paper_bgcolor=COLOR_NAVY,
                plot_bgcolor=COLOR_NAVY,
                height=500  # Increased height to show all 10 topics clearly
            )
            
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("Heatmap data unavailable.")
    
    st.divider()
    
    # Coalition Analytics
    st.subheader("Coalition Accountability Comparison")
    col_c1, col_c2 = st.columns([2, 1])
    
    with col_c1:
        coalition_stance = df_filtered.groupby(["Coalition", "Stance"]).size().reset_index(name="Count")
        if not coalition_stance.empty:
            fig_coal = px.bar(
                coalition_stance,
                x="Coalition",
                y="Count",
                color="Stance",
                color_discrete_map=STANCE_COLORS,
                barmode="group",
                title="Government vs Opposition Stance Profile",
                template="plotly_dark"
            )
            fig_coal.update_layout(paper_bgcolor=COLOR_NAVY, plot_bgcolor=COLOR_NAVY)
            st.plotly_chart(fig_coal, use_container_width=True)
        
    with col_c2:
        st.write("#### Key Findings")
        if not df_filtered.empty:
            evasive_pct = (df_filtered["Stance"] == "Evasive").mean() * 100
            st.metric("Thematic Evasiveness", f"{evasive_pct:.1f}%", delta="Parliamentary Avg")
            
            # Use stable heat_data if available
            try:
                most_evasive_topic = heat_data.mean(axis=1).idxmax()
                least_evasive_party = heat_data.mean(axis=0).idxmin()
            except:
                most_evasive_topic = "N/A"
                least_evasive_party = "N/A"
                
            st.markdown(f"""
            - **Most Evasive Topic:** {most_evasive_topic}
            - **Least Evasive Party:** {least_evasive_party}
            
            *Note: Analysis restricted to substantive thematic topics.*
            """)

with tab3:
    st.header("Hansard Evidence Explorer")
    st.markdown("---")
    
    # Search and Filter
    search_query = st.text_input("🔍 Search Substantive Speeches", placeholder="e.g. 'cukai', 'pendidikan', 'rasuah'...")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        party_options = sorted([str(x) for x in df_filtered["Party"].dropna().unique()])
        f_party = st.multiselect("Filter Party", options=party_options)
    with col_f2:
        stance_options = sorted([str(x) for x in df_filtered["Stance"].dropna().unique()])
        f_stance = st.multiselect("Filter Stance", options=stance_options)
    with col_f3:
        topic_options = sorted([str(x) for x in df_filtered["Topic_Label_Friendly"].dropna().unique()])
        f_topic = st.multiselect("Filter Topic", options=topic_options)

    # Apply filters
    df_search = df_filtered.copy()
    if search_query:
        df_search = df_search[df_search["Text"].str.contains(search_query, case=False, na=False)]
    if f_party:
        df_search = df_search[df_search["Party"].isin(f_party)]
    if f_stance:
        df_search = df_search[df_search["Stance"].isin(f_stance)]
    if f_topic:
        df_search = df_search[df_search["Topic_Label_Friendly"].isin(f_topic)]

    st.write(f"Showing **{len(df_search)}** substantive results")
    
    # Display table
    event = st.dataframe(
        df_search[["Date", "Speaker", "Party", "Topic_Label_Friendly", "Stance", "Text"]],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Detailed View (Speech Card)
    if len(event.selection.rows) > 0:
        selected_row_idx = event.selection.rows[0]
        selected_row = df_search.iloc[selected_row_idx]
        
        st.markdown("### Selected Speech Detail")
        
        stance_class = str(selected_row["Stance"]).lower()
        highlighted_text = highlight_text(selected_row['Text'], search_query)
        
        card_html = f"""<div class="speech-card {stance_class}-card">
<h4 style='color: {COLOR_GOLD}; margin-top: 0;'>{selected_row['Speaker']} ({selected_row['Party']})</h4>
<p style='color: {COLOR_GREY}; font-size: 0.9em;'>{selected_row['Date']} | {selected_row['Constituency'] or 'No Constituency'} | Topic: {selected_row['Topic_Label_Friendly']}</p>
<div style='background-color: {COLOR_NAVY}; padding: 15px; border-radius: 5px; margin: 15px 0;'>
{highlighted_text}
</div>
<div style='display: flex; justify-content: space-between; align-items: center;'>
<span style='background-color: {STANCE_COLORS.get(selected_row["Stance"], COLOR_GREY)}; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;'>
AI Stance: {selected_row['Stance']}
</span>
<span style='color: {COLOR_GREY}; font-style: italic;'>Analyzed by XLM-RoBERTa Intelligence</span>
</div>
</div>"""
        st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.info("👆 **Select a substantive row** to view the full speech and AI analysis.")

with tab4:
    st.header("Member of Parliament Insights")
    st.markdown("---")
    
    # Filter out noise from speaker list
    SPEAKER_NOISE = ["Antara ", "Beberapa ", "Seorang Ahli", "Tuan Pengerusi", "Tuan Yang di-Pertua", "Timbalan Yang di-Pertua", "Berdasarkan ", "Budaya menyatakan", "Menurut "]
    
    # Advanced filter: Speakers must be relatively short (names are usually < 60 chars) 
    # and not start with common sentence starters or lowercase letters
    speakers_list = [
        str(x) for x in df_master["Speaker"].dropna().unique() 
        if not any(noise in str(x) for noise in SPEAKER_NOISE)
        and len(str(x)) < 70
        and str(x)[0].isupper()
    ]
    all_speakers = sorted(speakers_list)
    
    col_mp1, col_mp2 = st.columns([1, 2])
    
    with col_mp1:
        st.subheader("Select MP")
        selected_mp = st.selectbox("Search/Select Member of Parliament", all_speakers, key="mp_selector")
        
        if selected_mp:
            mp_data = df_master[df_master["Speaker"] == selected_mp]
            mp_evasive_index = (mp_data["Stance"] == "Evasive").mean() * 100
            total_speeches = len(mp_data)
            
            # Party & Coalition info
            party = mp_data["Party"].iloc[0] if not mp_data["Party"].empty else "Unknown"
            coalition = mp_data["Coalition"].iloc[0] if not mp_data["Coalition"].empty else "Unknown"
            
            st.markdown(f"""<div style='background-color: {COLOR_NAVY}; padding: 20px; border-radius: 15px; border: 1px solid {COLOR_GOLD}44; text-align: center;'>
<h2 style='margin: 0; color: {COLOR_GOLD}; font-size: 1.5em;'>{selected_mp}</h2>
<p style='margin: 5px 0; color: {COLOR_WHITE}; opacity: 0.8;'>{party} | {coalition}</p>
<hr style='border: 0; border-top: 1px solid {COLOR_GOLD}22; margin: 15px 0;'>
<div style='display: flex; justify-content: space-around;'>
<div>
<p style='margin: 0; font-size: 0.8em; color: {COLOR_GREY};'>TOTAL SPEECHES</p>
<p style='margin: 0; font-size: 1.5em; font-family: "Playfair Display", serif; color: {COLOR_GOLD};'>{total_speeches}</p>
</div>
<div>
<p style='margin: 0; font-size: 0.8em; color: {COLOR_GREY};'>EVASIVE INDEX</p>
<p style='margin: 0; font-size: 1.5em; font-family: "Playfair Display", serif; color: {COLOR_AMBER};'>{mp_evasive_index:.1f}%</p>
</div>
</div>
</div>""", unsafe_allow_html=True)
            
            st.write("")
            st.write("**Top Focus Areas:**")
            # For MP Focus, we use the thematic labels
            top_mp_topics = mp_data[~mp_data["Topic_Int"].isin(NOISE_TOPICS)]["Topic_Label_Friendly"].value_counts().head(5)
            
            if not top_mp_topics.empty:
                # Build HTML without leading spaces to prevent markdown code block detection
                chips_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
                for t, c in top_mp_topics.items():
                    chips_html += f"<div style='background-color: {COLOR_NAVY}; border: 1px solid {COLOR_GOLD}66; padding: 5px 12px; border-radius: 20px; font-size: 0.85em;'>"
                    chips_html += f"<span style='color: {COLOR_GOLD}; font-weight: bold;'>{c}</span>"
                    chips_html += f"<span style='color: {COLOR_WHITE}; opacity: 0.9;'> {t}</span></div>"
                chips_html += "</div>"
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.markdown("*Procedural/Administrative focus*")

    with col_mp2:
        if selected_mp:
            st.subheader("Accountability Profile")
            
            # Stance Distribution
            stance_counts = mp_data["Stance"].value_counts().reset_index(name="Count")
            fig_mp_stance = px.pie(
                stance_counts, 
                values="Count", 
                names="Stance", 
                color="Stance",
                color_discrete_map=STANCE_COLORS,
                hole=0.4,
                template="plotly_dark",
                title=f"Stance Distribution: {selected_mp}"
            )
            fig_mp_stance.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_mp_stance, use_container_width=True)
            
            st.subheader("Key Speech Evidence")
            # Filter mp_data to exclude noise for evidence
            mp_data_clean = mp_data[~mp_data["Topic_Int"].isin(NOISE_TOPICS)]
            
            # Show a few representative speeches (one evasive, one pro/con)
            evasive_speeches = mp_data_clean[mp_data_clean["Stance"] == "Evasive"].head(1)
            other_speeches = mp_data_clean[mp_data_clean["Stance"] != "Evasive"].head(1)
            
            for _, row in pd.concat([evasive_speeches, other_speeches]).iterrows():
                stance_class = str(row["Stance"]).lower()
                st.markdown(f"""<div class="speech-card {stance_class}-card" style='padding: 15px; margin-bottom: 10px;'>
<p style='font-size: 0.8em; color: {COLOR_GREY}; margin-bottom: 5px;'>{row['Date']} | Topic: {row['Topic_Label_Friendly']}</p>
<p style='font-size: 0.95em; margin-bottom: 10px;'>"{row['Text'][:300]}..."</p>
<span style='background-color: {STANCE_COLORS.get(row["Stance"], COLOR_GREY)}; color: white; padding: 2px 10px; border-radius: 10px; font-size: 0.8em;'>
{row['Stance']}
</span>
</div>""", unsafe_allow_html=True)

with tab5:
    st.header("Project Intelligence & Methodology")
    st.markdown("---")
    
    m_col1, m_col2 = st.columns([2, 1])
    
    with m_col1:
        st.subheader("The AI Pipeline")
        st.markdown(f"""
        This project bridges the **'Hansard Gap'** using advanced Natural Language Processing to analyze parliamentary accountability in the Malaysian Dewan Rakyat.
        
        #### 1. Data Ingestion & Parsing
        - **Source:** 29 PDF Hansard records (2018-2022).
        - **Engine:** Custom state-machine parser using `pdfplumber` to extract over **16,000 unique speech turns**.
        - **Metadata:** Speeches are enriched with Party and Coalition data via fuzzy matching against parliamentary member lists.
        
        #### 2. Topic Discovery (BERTopic)
        - **Model:** `paraphrase-multilingual-MiniLM-L12-v2` embedding layer.
        - **Technique:** Unsupervised clustering to identify thematic trends without human bias.
        - **Validation:** Achieved a **Topic Coherence (Cv) of 0.3909**, successfully isolating substantive policy from procedural 'noise'.
        
        #### 3. Stance & Evasiveness (XLM-RoBERTa)
        - **Model:** Fine-tuned `xlm-roberta-base`.
        - **Training:** Supervised learning on 'Silver Labels' generated via LLM-augmented grounding.
        - **Classes:** `Pro`, `Con`, `Neutral`, and the critical `Evasive` tag.
        - **Performance:** **69.1% overall accuracy**, specifically optimized to distinguish between neutral replies and evasive non-answers.
        """)
        
    with m_col2:
        st.subheader("Technical Specs")
        st.markdown(f"""<div style='background-color: {COLOR_NAVY}; padding: 20px; border-radius: 10px; border: 1px solid {COLOR_GOLD};'>
<p style='color: {COLOR_GOLD}; font-weight: bold; margin-bottom: 5px;'>DEVELOPMENT STACK</p>
<ul style='font-size: 0.9em; color: {COLOR_WHITE};'>
<li>Streamlit Architecture</li>
<li>Plotly Data Visualization</li>
<li>Transformers (HuggingFace)</li>
<li>Scikit-learn / Pandas</li>
</ul>
<p style='color: {COLOR_GOLD}; font-weight: bold; margin-top: 15px; margin-bottom: 5px;'>DATA VOLUME</p>
<p style='font-size: 1.2em; font-family: "Playfair Display", serif;'>16,050 Speeches</p>
<p style='color: {COLOR_GOLD}; font-weight: bold; margin-top: 15px; margin-bottom: 5px;'>VERSION</p>
<p style='font-size: 1.2em;'>v1.0.0 (FYP Prototype)</p>
</div>""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7F8C8D; font-size: 0.8em;'>
        Malaysian Hansard Parliamentary Analytics Dashboard &copy; 2026 | Developed for Final Year Project
    </div>
    """,
    unsafe_allow_html=True
)
