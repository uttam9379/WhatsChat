import streamlit as st
import pandas as pd
import re
from collections import Counter
import emoji
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO
from pathlib import Path
import fitz  # PyMuPDF
import docx
import os
import exifread
import wave
import contextlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# Set page config
st.set_page_config(
    page_title="WhatsChat Analyzer",
    layout="wide",
    page_icon="icon.png"  
)

# Load and display logo in sidebar
logo = Image.open("icon.png")
st.sidebar.image(logo, caption="WhatsChat Analyzer", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Created by **Uttam Kumar Das**")



st.sidebar.subheader("📎 Media File Analysis")
media_folder = st.sidebar.file_uploader("📂 Upload WhatsApp Chat include media (.ZIP) only", type="zip")

if media_folder:
    import zipfile
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(media_folder, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        media_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                full_path = os.path.join(root, file)
                ext = file.lower().split('.')[-1]
                if ext in ['jpg', 'jpeg', 'png', 'gif', 'mp4', 'avi', 'pdf', 'docx']:
                    media_files.append(full_path)

        if media_files:
            st.markdown("### 📁 Extracted Media Files")
            for file in media_files:
                ext = file.lower().split('.')[-1]
                st.markdown(f"**📎 {Path(file).name}**")

                if ext in ['jpg', 'jpeg', 'png', 'gif']:
                    img = Image.open(file)
                    st.image(img, caption=Path(file).name, use_container_width=True)

                elif ext in ['mp4', 'avi']:
                    with open(file, "rb") as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)

                elif ext == 'pdf':
                    st.write("📄 PDF Preview:")
                    with fitz.open(file) as pdf_doc:
                     for page_num in range(min(2, len(pdf_doc))):  # show first 2 pages
                        page = pdf_doc.load_page(page_num)
                        pix = page.get_pixmap()
                        img_bytes = pix.tobytes("png")
                        st.image(img_bytes, caption=f"{Path(file).name} - Page {page_num+1}")


                elif ext == 'docx':
                    doc = docx.Document(file)
                    content = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
                    st.text_area("📄 Word Document Content", value=content, height=200)
        else:
            st.warning("No supported media files found in the ZIP.")




# Theme toggle in sidebar
theme = st.sidebar.selectbox("Choose Theme:", ["Dark", "Light"])

# Apply CSS based on theme
if theme == "Dark":
    st.markdown("""
        <style>
        body { background-color: #0e1117; color: white; }
        .css-1d391kg { background-color: #1e1e1e; }
        .stApp { background-color: #0e1117; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body { background-color: white; color: black; }
        .css-1d391kg { background-color: #f0f0f0; }
        .stApp { background-color: white; }
        </style>
    """, unsafe_allow_html=True)


# Load image
logo = Image.open("icon.png")
# Display title and icon side by side using HTML + base64
buffered = BytesIO()
logo.save(buffered, format="PNG")
img_b64 = base64.b64encode(buffered.getvalue()).decode()

st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px;">
        <img src="data:image/png;base64,{img_b64}" alt="Logo" width="50"/>
        <h1 style="margin: 0;">WhatsChat Analyzer</h1>
    </div>
""", unsafe_allow_html=True)

def extract_emojis(text):
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)

def load_chat(file):
    chat = file.read().decode("utf-8")
    messages = []

    pattern_12hr = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}\s?[APap][Mm]) - (.*)$')
    pattern_24hr = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{2}:\d{2}) - (.*)$')

    is_12hr = False
    for line in chat.splitlines():
        if pattern_12hr.match(line):
            is_12hr = True
            break
        elif pattern_24hr.match(line):
            break

    pattern = pattern_12hr if is_12hr else pattern_24hr

    for line in chat.splitlines():
        match = pattern.match(line)
        if match:
            date, time_str, content = match.groups()
            if ': ' in content:
                sender, message = content.split(': ', 1)
            else:
                sender, message = "System", content
            messages.append([date, time_str, sender, message])

    df = pd.DataFrame(messages, columns=["Date", "Time", "Sender", "Message"])

    def parse_datetime(row):
        date_str = row["Date"]
        time_str = row["Time"].strip()
        try:
            if is_12hr:
                return datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %I:%M %p")
            else:
                return datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
        except:
            return pd.NaT

    df["Datetime"] = df.apply(parse_datetime, axis=1)
    df["Date"] = df["Datetime"].dt.date
    df["Time"] = df["Datetime"].dt.time
    df.drop(columns=["Datetime"], inplace=True)
    df.dropna(inplace=True)
    return df.copy(), df, is_12hr

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

st.subheader("📎Chat File Analysis")

uploaded_file = st.file_uploader("📁 Upload WhatsApp Chat (.txt) only", type=["txt"])

if uploaded_file:
    df, original_df, is_12hr = load_chat(uploaded_file)

    st.sidebar.header("📊 Options")
    st.sidebar.info(f"Detected Time Format: {'12-hour (AM/PM)' if is_12hr else '24-hour'}")
    show_system = st.sidebar.checkbox("Include System Messages", value=True)

    if not show_system:
        df = df[df["Sender"] != "System"]

    df["Sentiment"] = df["Message"].apply(analyze_sentiment)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 INFO", "📊 STATS", "📈 GRAPHS", "🔗 LINKS", "🧠 ADVANCE"])


    # ----------------- INFO TAB -----------------
    with tab1:
        st.subheader("📌 Chat Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Total Messages:", df.shape[0])
            st.write("Participants:", ', '.join(df['Sender'].unique()))
            st.write("Most Active:", df['Sender'].value_counts().idxmax() if not df['Sender'].empty else "N/A")
            top_emojis = Counter(''.join(df['Message'].apply(extract_emojis))).most_common(5)
            st.write("Top Emojis:", ' '.join([e[0] for e in top_emojis]))

        with col2:
            st.write("Average Sentiment Score:", round(df["Sentiment"].mean(), 3))
            st.write("Positive Messages:", (df["Sentiment"] > 0).sum())
            st.write("Negative Messages:", (df["Sentiment"] < 0).sum())
            st.write("Neutral Messages:", (df["Sentiment"] == 0).sum())

        with st.expander("📋 System Message Summary"):
            sys_msgs = original_df[original_df["Sender"] == "System"]
            st.write(f"Total system messages: {sys_msgs.shape[0]}")
            if not sys_msgs.empty:
                st.dataframe(sys_msgs['Message'].value_counts().head(10))
            else:
                st.write("No system messages found.")

    # ----------------- STATS TAB -----------------
    with tab2:
        st.subheader("📊 Chat Statistics")
        stat_sender = st.selectbox("Filter stats by sender:", options=["Everyone"] + list(df["Sender"].unique()))
        stat_df = df if stat_sender == "Everyone" else df[df["Sender"] == stat_sender]

        emoji_counts = Counter(''.join(stat_df['Message'].apply(extract_emojis)))
        if emoji_counts:
            top_emoji_df = pd.DataFrame(emoji_counts.most_common(10), columns=['Emoji', 'Count'])
            st.write("😊 Top Emojis:")
            st.dataframe(top_emoji_df)
        else:
            st.write("No emojis found.")

        st.write("✉️ Total Messages Sent:", stat_df.shape[0])
        total_words = stat_df["Message"].apply(lambda x: len(str(x).split())).sum()
        st.write("📝 Total Words Sent:", total_words)

    # ----------------- GRAPHS TAB -----------------
    with tab3:
        st.subheader("📈 Visualizations and Graphs")
        st.markdown("### 🔍 Message Filter Section")
        user_filter = st.multiselect("Filter by sender:", options=df['Sender'].unique(), default=list(df['Sender'].unique()))
        keyword = st.text_input("Search keyword (optional):")

        filtered = df[df["Sender"].isin(user_filter)]
        if keyword:
            filtered = filtered[filtered["Message"].str.contains(keyword, case=False, na=False)]

        st.write(f"Showing {len(filtered)} filtered messages:")
        st.dataframe(filtered[['Date', 'Time', 'Sender', 'Message', 'Sentiment']])

        st.markdown("### 📅 Messages Over Time")
        msgs_per_day = filtered.groupby('Date').size()
        fig, ax = plt.subplots()
        sns.lineplot(x=msgs_per_day.index, y=msgs_per_day.values, ax=ax, color='cyan' if theme == "Dark" else 'blue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Messages")
        if theme == "Dark":
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        st.pyplot(fig)

        st.markdown("### 📊 Sentiment Distribution")
        sentiment_counts = pd.cut(filtered["Sentiment"], bins=[-1, -0.01, 0.01, 1], labels=["Negative", "Neutral", "Positive"]).value_counts()
        fig2, ax2 = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=['red', 'gray', 'green'], ax=ax2)
        ax2.set_ylabel("Number of Messages")
        ax2.set_xlabel("Sentiment")
        if theme == "Dark":
            ax2.set_facecolor('#0e1117')
            fig2.patch.set_facecolor('#0e1117')
            ax2.tick_params(colors='white')
            ax2.xaxis.label.set_color('white')
            ax2.yaxis.label.set_color('white')
        st.pyplot(fig2)

        st.markdown("### 👥 Most Active Participants")
        active_counts = filtered['Sender'].value_counts().head(10)
        fig3, ax3 = plt.subplots()
        sns.barplot(x=active_counts.values, y=active_counts.index, palette='viridis', ax=ax3)
        ax3.set_xlabel("Number of Messages")
        ax3.set_ylabel("Sender")
        if theme == "Dark":
            ax3.set_facecolor('#0e1117')
            fig3.patch.set_facecolor('#0e1117')
            ax3.tick_params(colors='white')
            ax3.xaxis.label.set_color('white')
            ax3.yaxis.label.set_color('white')
        st.pyplot(fig3)

        st.markdown("### 😊 Top Emojis")
        emoji_counts = Counter(''.join(filtered['Message'].apply(extract_emojis)))
        if emoji_counts:
            emojies, counts = zip(*emoji_counts.most_common(20))
            emoji_df = pd.DataFrame({'Emoji': emojies, 'Count': counts})
            st.dataframe(emoji_df)
        else:
            st.write("No emojis found.")

        st.markdown("### ☁️ Word Cloud")
        text_for_wc = ' '.join(filtered['Message'].astype(str).tolist())
        if text_for_wc.strip():
            wc = WordCloud(width=800, height=400, background_color='black' if theme == "Dark" else 'white',
                        colormap='Pastel1').generate(text_for_wc)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            if theme == "Dark":
                fig_wc.patch.set_facecolor('#0e1117')
            st.pyplot(fig_wc)
        else:
            st.write("No text found for Word Cloud.")

        st.markdown("### 📅 Sentiment Timeline per User")
        if filtered.empty:
            st.write("No data to show sentiment timeline.")
        else:
            fig_st, ax_st = plt.subplots(figsize=(12, 6))
            for user in user_filter:
                user_df = filtered[filtered['Sender'] == user]
                if not user_df.empty:
                    user_sentiment = user_df.copy()
                    user_sentiment["Date"] = pd.to_datetime(user_sentiment["Date"])
                    user_sentiment = user_sentiment.set_index('Date')['Sentiment'].resample('W').mean()
                    ax_st.plot(user_sentiment.index, user_sentiment.values, marker='o', label=user)
            ax_st.set_xlabel("Date")
            ax_st.set_ylabel("Average Sentiment")
            ax_st.legend()
            ax_st.axhline(0, color='gray', linestyle='--', linewidth=1)
            if theme == "Dark":
                ax_st.set_facecolor('#0e1117')
                fig_st.patch.set_facecolor('#0e1117')
                ax_st.tick_params(colors='white')
                ax_st.xaxis.label.set_color('white')
                ax_st.yaxis.label.set_color('white')
                ax_st.legend(facecolor='#0e1117', edgecolor='white', labelcolor='white')
            st.pyplot(fig_st)

    # ----------------- LINKS TAB -----------------
    with tab4:
        st.subheader("🔗 Most Sent Links")
        sender_link = st.selectbox("Select sender:", options=["Everyone"] + list(df["Sender"].unique()))
        link_df = df if sender_link == "Everyone" else df[df["Sender"] == sender_link]
        link_pattern = r'(https?://\S+)'  # Capture group
        all_links = link_df["Message"].str.extractall(link_pattern).dropna()
        link_counts = all_links[0].value_counts().head(20)
        if not link_counts.empty:
            st.dataframe(link_counts.rename("Count").reset_index().rename(columns={"index": "Link"}))
        else:
            st.write("No links found.")

            # --- CHAT ANALYSIS ADDITIONS ---
            #---- ADVANCE TAB----
if uploaded_file:
 with tab5:



    st.subheader("## 🔍 Advanced Chat Insights")

    # 1. Activity Heatmap
    st.markdown("### 📆 Activity Heatmap (Day vs Hour)")
    heatmap_df = df.copy()
    heatmap_df["Hour"] = pd.to_datetime(heatmap_df["Time"].astype(str)).dt.hour
    heatmap_df["Weekday"] = pd.to_datetime(heatmap_df["Date"].astype(str)).dt.day_name()
    pivot = pd.pivot_table(heatmap_df, index="Weekday", columns="Hour", values="Message", aggfunc="count").fillna(0)
    fig_hm, ax_hm = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, cmap="YlGnBu", ax=ax_hm)
    st.pyplot(fig_hm)

    # 2. Response Time Analysis
    st.markdown("### ⏱️ Response Time Analysis")
    df_sorted = df.sort_values(by=["Date", "Time"])
    df_sorted["Next_Sender"] = df_sorted["Sender"].shift(-1)
    df_sorted["Next_Time"] = pd.to_datetime(df_sorted["Date"].astype(str) + " " + df_sorted["Time"].astype(str)).shift(-1)
    df_sorted["Current_Time"] = pd.to_datetime(df_sorted["Date"].astype(str) + " " + df_sorted["Time"].astype(str))
    df_sorted["Response_Time"] = (df_sorted["Next_Time"] - df_sorted["Current_Time"]).dt.total_seconds()
    df_sorted = df_sorted[df_sorted["Sender"] != df_sorted["Next_Sender"]]
    st.write(f"Average response time: {round(df_sorted['Response_Time'].mean()/60, 2)} minutes")

    # 3. Keyword and Topic Clustering
    st.markdown("### 🧠 Keyword Clustering")
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df["Message"])
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    df["Cluster"] = kmeans.labels_
    st.write(df[["Sender", "Message", "Cluster"]].head())

    # 4. Cumulative Message Count
    st.markdown("### 📈 Cumulative Message Count")
    df['Datetime'] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    cumulative = df.groupby("Datetime").size().cumsum()
    fig_cum, ax_cum = plt.subplots()
    ax_cum.plot(cumulative.index, cumulative.values, color='green')
    ax_cum.set_xlabel("Time")
    ax_cum.set_ylabel("Cumulative Messages")
    st.pyplot(fig_cum)

    

    # 6. Pinned Message Detection
    st.markdown("### 📌 Pinned Messages")
    pinned_df = df[df["Message"].str.contains("pinned message", case=False, na=False)]
    if not pinned_df.empty:
        st.dataframe(pinned_df[["Sender", "Date", "Message"]])
    else:
        st.write("No pinned messages found.")

    # 7. Most Replied Messages (Basic)
    st.markdown("### 🔁 Most Replied Messages (Pattern-Based)")
    replies = df["Message"].str.extract(r'@(.+?):')
    reply_counts = replies[0].value_counts().head(5)
    st.write(reply_counts.rename("Reply Count"))

            # ===================== ADDITIONAL FEATURES (AFTER ORIGINAL CODE) =====================

            # Ensure media_files is defined
media_files = globals().get("media_files", [])


# --- MEDIA ANALYSIS ADDITIONS ---
if media_files:
    st.markdown("## 🧠 Additional Media Insights")

    # 1. EXIF Metadata Extraction
    st.markdown("### 🧾 EXIF Metadata (Images)")
    for file in media_files:
        if file.lower().endswith(('jpg', 'jpeg')):
            st.markdown(f"**📷 {Path(file).name}**")
            if os.path.exists(file):
                with open(file, 'rb') as f:
                     tags = exifread.process_file(f, stop_tag="UNDEF", details=False)
                     if tags:
                        st.json({tag: str(tags[tag]) for tag in list(tags)[:5]})  # limit to 5 tags
                     else:
                        st.write("No EXIF data found.")
                        
                        tags = exifread.process_file(f, stop_tag="UNDEF", details=False)
                if tags:
                    st.json({tag: str(tags[tag]) for tag in list(tags)[:5]})  # limit to 5 tags
                else:
                    st.write("No EXIF data found.")

   

    
    # 5. Basic Audio Analysis
    st.markdown("### 🔊 Audio File Info")
    for file in media_files:
        if file.endswith(".wav"):
            with contextlib.closing(wave.open(file, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                st.write(f"**{Path(file).name}**: {round(duration, 2)}s | {f.getnchannels()} channels | {rate} Hz")


                
                


