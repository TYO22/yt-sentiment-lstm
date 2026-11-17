import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from src.scraper import extract_video_id, fetch_youtube_data
from src.predictor import load_artifacts, predict_sentiments

st.set_page_config(
    page_title="YouTube Sentiment Analyzer (Bi-LSTM)",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def get_model_bundle():
    return load_artifacts()

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_video(url: str, api_key: str, max_comments: int, sort_by: str):
    model, tokenizer, le = get_model_bundle()
    metadata, raw_comments = fetch_youtube_data(url, api_key=api_key, max_comments=max_comments, sort_by=sort_by)
    if not raw_comments:
        return metadata, pd.DataFrame()
    df_pred = predict_sentiments(raw_comments, model, tokenizer, le)
    return metadata, df_pred

# Ambil API Key dari secrets
try:
    API_KEY = st.secrets["YOUTUBE_API_KEY"]
except Exception:
    API_KEY = None

with st.sidebar:
    st.header("Pengaturan Analisis")
    max_comments = st.slider("Jumlah Maksimal Komentar", min_value=50, max_value=1000, value=200, step=50)
    sort_option = st.selectbox(
        "Urutkan Komentar Berdasarkan",
        options=["popular", "recent"],
        format_func=lambda x: "Terpopuler (Relevan)" if x == "popular" else "Terbaru"
    )
    st.divider()
    st.caption("Engine: Bidirectional LSTM + FastText Indonesia (300d)")

st.title("🎬 Analisis Sentimen Komentar YouTube")
st.markdown("Masukkan tautan video YouTube untuk menganalisis opini penonton secara otomatis menggunakan model **Bidirectional LSTM**.")

url_input = st.text_input("Tautan Video YouTube", placeholder="https://www.youtube.com/watch?v=...")
analyze_btn = st.button("Mulai Analisis", type="primary")

if analyze_btn:
    if not API_KEY:
        st.error("YOUTUBE_API_KEY belum dikonfigurasi pada secrets.toml / Streamlit Secrets.")
    elif not url_input.strip():
        st.warning("Silakan masukkan tautan video YouTube terlebih dahulu.")
    else:
        video_id = extract_video_id(url_input)
        if not video_id:
            st.error("Tautan YouTube tidak dikenali. Periksa kembali format URL-nya.")
        else:
            with st.spinner("Mengambil data video & menjalankan prediksi Bi-LSTM..."):
                try:
                    metadata, df_result = analyze_video(url_input, API_KEY, max_comments, sort_option)
                except Exception as e:
                    st.error(str(e))
                    st.stop()

            if df_result.empty:
                st.info("Tidak ditemukan komentar teks yang valid pada video ini.")
                st.stop()

            # Preview Video, Metadata & Ringkasan KPI
            col_vid, col_kpi = st.columns([1, 2])
            with col_vid:
                st.video(f"https://www.youtube.com/watch?v={video_id}")

            total = len(df_result)
            counts = df_result["sentiment"].value_counts()
            pos_count = counts.get("positive", 0)
            neu_count = counts.get("neutral", 0)
            neg_count = counts.get("negative", 0)
            dominant = counts.idxmax().capitalize()

            with col_kpi:
                st.subheader(metadata["title"])
                st.caption(f"Channel: **{metadata['channel']}** • {metadata['views']:,} kali ditonton")
                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Dianalisis", f"{total}")
                m2.metric("Positif", f"{pos_count} ({pos_count/total*100:.1f}%)")
                m3.metric("Netral", f"{neu_count} ({neu_count/total*100:.1f}%)")
                m4.metric("Negatif", f"{neg_count} ({neg_count/total*100:.1f}%)")
                st.success(f"Sentimen Dominan: **{dominant}**")

            st.divider()

            # Visualisasi Grafik
            color_map = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Proporsi Sentimen")
                fig_pie = px.pie(
                    df_result,
                    names="sentiment",
                    color="sentiment",
                    color_discrete_map=color_map,
                    hole=0.45
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                st.subheader("Distribusi Tingkat Keyakinan Model (%)")
                fig_hist = px.histogram(
                    df_result,
                    x="confidence",
                    color="sentiment",
                    color_discrete_map=color_map,
                    nbins=20
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # WordCloud per Sentimen
            st.subheader("☁️ Kata Kunci Dominan (WordCloud)")

            factory = StopWordRemoverFactory()
            stopwords_id = set(factory.get_stop_words())
            stopwords_id.update({"bang", "abang", "gue", "gw", "sih", "deh", "dong", "nih", "tuh", "aja", "nya"})

            wc_col1, wc_col2 = st.columns(2)
            wc_configs = [
                (wc_col1, "positive", "Komentar Positif", "Greens"),
                (wc_col2, "negative", "Komentar Negatif", "Reds"),
            ]

            for col, sentiment_label, title, cmap in wc_configs:
                with col:
                    st.markdown(f"**{title}**")
                    subset_text = " ".join(df_result[df_result["sentiment"] == sentiment_label]["clean_text"])
                    if subset_text.strip():
                        try: 
                            wc = WordCloud(
                                width=600,
                                height=350,
                                background_color="white",
                                colormap=cmap,
                                stopwords=stopwords_id
                            ).generate(subset_text)
                            
                            fig, ax = plt.subplots(figsize=(6, 3.5))
                            ax.imshow(wc, interpolation="bilinear")
                            ax.axis("off")
                            st.pyplot(fig)
                        except ValueError:
                            st.caption("Kosakata belum cukup beragam untuk membuat WordCloud.")
                    else:
                        st.caption(f"Tidak ada data {title.lower()} yang cukup.")

            st.divider()

            # Filter Tabel & Unduh Hasil
            st.subheader("Detail Komentar & Prediksi")
            selected_filter = st.multiselect(
                "Filter berdasarkan sentimen:",
                options=["positive", "neutral", "negative"],
                default=["positive", "neutral", "negative"]
            )
            filtered_df = df_result[df_result["sentiment"].isin(selected_filter)]

            st.dataframe(
                filtered_df[["author", "time", "text", "sentiment", "confidence", "likes"]],
                use_container_width=True
            )

            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Unduh Hasil Analisis (CSV)",
                data=csv_data,
                file_name=f"sentiment_{video_id}.csv",
                mime="text/csv"
            )