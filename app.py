
import streamlit as st
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from readability_features import (
    predict_readability, extract_stats
)

ROOT = Path(__file__).parent

st.set_page_config(
    page_title="CommonLit Readability",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(
        ROOT / "commonlit_final_model.joblib"
    )

bundle = load_model()

st.title("📚 CommonLit Readability Predictor")

st.write(
    "İngilizce metinlerin okunabilirlik "
    "puanını NLP ve Ridge Regression ile tahmin et."
)

c1, c2, c3 = st.columns(3)

c1.metric("Validation RMSE", "0.655027")
c2.metric("Validation MAE", "0.521909")
c3.metric("Model", "TF-IDF + Ridge")

st.info(
    "Bu uygulama eğitim amaçlıdır. "
    "Puanlar kesin okuma seviyesi veya "
    "öğrenci başarısı ölçümü değildir."
)

tab1, tab2 = st.tabs([
    "Metin Analizi",
    "CSV ile Toplu Tahmin"
])

with tab1:

    example = (
        "The curious child explored the forest "
        "and discovered many plants and animals. "
        "Each new discovery inspired another question."
    )

    text = st.text_area(
        "İngilizce metnini yaz veya yapıştır",
        value=example,
        height=220
    )

    if st.button(
        "Okunabilirlik Puanını Tahmin Et",
        type="primary"
    ):

        if len(text.strip()) < 20:
            st.warning(
                "Lütfen daha uzun bir metin gir."
            )

        else:
            score = float(
                predict_readability(
                    bundle, [text]
                )[0]
            )

            stats = extract_stats([text])[0]

            st.metric(
                "Tahmini okunabilirlik puanı",
                f"{score:.4f}"
            )

            st.caption(
                "Bu puan CommonLit hedef ölçeğindedir. "
                "Flesch veya okul sınıfı puanı değildir."
            )

            a, b, c = st.columns(3)

            a.metric(
                "Kelime sayısı",
                int(stats[0])
            )

            b.metric(
                "Cümle sayısı",
                int(stats[1])
            )

            c.metric(
                "Ortalama kelime uzunluğu",
                f"{stats[3]:.2f}"
            )

            st.write(
                "Flesch Reading Ease (yaklaşık):",
                f"{stats[17]:.2f}"
            )

with tab2:

    st.subheader("Toplu metin tahmini")

    st.write(
        "CSV dosyasında `excerpt` sütunu "
        "bulunmalıdır. `id` sütunu isteğe bağlıdır."
    )

    uploaded = st.file_uploader(
        "CSV dosyasını seç",
        type=["csv"]
    )

    if uploaded is not None:

        df = pd.read_csv(
            uploaded,
            nrows=2001
        )

        if "excerpt" not in df.columns:
            st.error(
                "CSV dosyasında excerpt sütunu yok."
            )

        elif not 1 <= len(df) <= 2000:
            st.error(
                "En fazla 2.000 metin yükleyebilirsin."
            )

        else:

            st.write("Yüklenen metin:", len(df))

            if st.button(
                "Toplu Tahmin Yap",
                type="primary"
            ):
                texts = (
                    df["excerpt"]
                    .fillna("")
                    .astype(str)
                    .tolist()
                )

                predictions = predict_readability(
                    bundle,
                    texts
                )

                result = pd.DataFrame({
                    "id": (
                        df["id"].astype(str)
                        if "id" in df.columns
                        else [
                            f"row_{i+1}"
                            for i in range(len(df))
                        ]
                    ),
                    "predicted_target": predictions
                })

                st.success(
                    f"{len(result)} metin analiz edildi."
                )

                st.dataframe(
                    result.head(30),
                    hide_index=True
                )

                st.download_button(
                    "Tahminleri CSV indir",
                    data=result.to_csv(index=False),
                    file_name="readability_predictions.csv",
                    mime="text/csv"
                )

st.caption(
    "Kaggle CommonLit Readability Prize | "
    "Educational NLP project"
)
