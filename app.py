import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Configuración inicial
# ============================================================

st.set_page_config(
    page_title="Recomendador YouTube",
    page_icon="📈",
    layout="wide"
)

DAY_NAMES_ES = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo"
}

DAY_ORDER = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


# ============================================================
# Carga de artifacts
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("artifacts/ridge_model.joblib")


@st.cache_data
def load_config():
    with open("artifacts/feature_config.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_data():
    df = pd.read_csv("artifacts/df_fe.csv")

    df["published_at_local"] = pd.to_datetime(
        df["published_at_local"],
        utc=True,
        errors="coerce"
    ).dt.tz_convert("Europe/Madrid")

    if "is_short" in df.columns:
        if df["is_short"].dtype == "object":
            df["is_short"] = (
                df["is_short"]
                .astype(str)
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0})
                .fillna(0)
                .astype(int)
            )
        else:
            df["is_short"] = df["is_short"].astype(int)

    if "log_views_7d_est" not in df.columns and "views_7d_est" in df.columns:
        df["log_views_7d_est"] = np.log1p(df["views_7d_est"])

    return df


model = load_model()
config = load_config()
df_fe = load_data()

all_feature_cols = config["all_feature_cols"]
categorical_features = config["categorical_features"]
target_col = config["target_col"]

for col in categorical_features:
    if col in df_fe.columns:
        df_fe[col] = df_fe[col].fillna("Unknown").astype(str)


# ============================================================
# Funciones auxiliares para entrada manual
# ============================================================

def count_tags_from_text(tags_text):
    if pd.isna(tags_text):
        return 0

    tags_text = str(tags_text).strip()

    if tags_text == "":
        return 0

    if "," in tags_text:
        return len([tag for tag in tags_text.split(",") if tag.strip() != ""])

    return len(tags_text.split())


def build_manual_base_video_row(
    df_history,
    channel_title,
    is_short,
    title,
    description,
    tags_text,
    duration_minutes,
    category_id,
    channel_country,
    definition,
    caption
):
    """
    Construye una fila base para un vídeo introducido manualmente.

    No usa views, likes, comentarios ni target.
    Solo construye variables conocidas antes de publicar.
    Las variables temporales e históricas se recalculan después para cada horario candidato.
    """

    channel_rows = (
        df_history[df_history["channel_title"] == channel_title]
        .sort_values("published_at_local")
        .copy()
    )

    if len(channel_rows) == 0:
        raise ValueError("El canal seleccionado no existe en el histórico.")

    channel_id = channel_rows["channel_id"].iloc[-1]

    same_format_rows = channel_rows[channel_rows["is_short"] == int(is_short)]

    if len(same_format_rows) > 0:
        reference_row = same_format_rows.iloc[-1]
    else:
        reference_row = channel_rows.iloc[-1]

    row = {}

    # Inicializar todas las features del modelo
    for col in all_feature_cols:
        row[col] = np.nan

    # Identificación auxiliar
    row["channel_id"] = channel_id
    row["channel_title"] = channel_title
    row["title"] = title

    # Formato
    row["is_short"] = int(is_short)

    # Duración
    row["duration_minutes"] = float(duration_minutes)
    row["log_duration_minutes"] = np.log1p(float(duration_minutes))

    if "duration_seconds" in all_feature_cols:
        row["duration_seconds"] = float(duration_minutes) * 60

    # Caption
    row["caption"] = int(caption)

    # Título
    title_clean = "" if pd.isna(title) else str(title)

    row["title_length"] = len(title_clean)
    row["title_word_count"] = len(title_clean.split())
    row["title_has_question"] = int("?" in title_clean)
    row["title_has_exclamation"] = int("!" in title_clean)

    # Descripción
    description_clean = "" if pd.isna(description) else str(description)

    row["has_description"] = int(description_clean.strip() != "")
    row["description_length"] = len(description_clean)
    row["description_word_count"] = len(description_clean.split())

    # Tags
    tags_count = count_tags_from_text(tags_text)

    row["tags_count"] = tags_count
    row["has_tags"] = int(tags_count > 0)

    # Categóricas
    row["category_id"] = str(category_id)
    row["channel_country"] = str(channel_country)
    row["definition"] = str(definition)

    # Para cualquier feature que exista en el modelo y no se haya rellenado,
    # usamos como referencia el último vídeo del mismo canal/formato.
    for col in all_feature_cols:
        if pd.isna(row.get(col, np.nan)) and col in reference_row.index:
            row[col] = reference_row[col]

    return pd.Series(row)


# ============================================================
# Funciones del recomendador
# ============================================================

def generate_candidate_slots(start_datetime, days_ahead=7):
    candidate_times = pd.date_range(
        start=start_datetime,
        periods=days_ahead * 24,
        freq="1h"
    )

    return pd.DataFrame({
        "candidate_published_at_local": candidate_times
    })


def compute_history_for_candidate(
    df_history,
    channel_id,
    is_short,
    candidate_time,
    h_days=7,
    window_size=10
):
    channel_history = (
        df_history[
            (df_history["channel_id"] == channel_id) &
            (df_history["published_at_local"] < candidate_time)
        ]
        .sort_values("published_at_local")
        .copy()
    )

    format_history = (
        channel_history[channel_history["is_short"] == int(is_short)]
        .sort_values("published_at_local")
        .copy()
    )

    channel_posts_before = len(channel_history)
    channel_format_posts_before = len(format_history)

    if len(channel_history) > 0:
        last_post_time = channel_history["published_at_local"].max()
        hours_since_last_post = (candidate_time - last_post_time).total_seconds() / 3600
        has_previous_post = 1
    else:
        hours_since_last_post = np.nan
        has_previous_post = 0

    if len(format_history) > 0:
        last_format_post_time = format_history["published_at_local"].max()
        hours_since_last_format_post = (candidate_time - last_format_post_time).total_seconds() / 3600
        has_previous_format_post = 1
    else:
        hours_since_last_format_post = np.nan
        has_previous_format_post = 0

    def count_posts_last_hours(history_df, hours):
        start_time = candidate_time - pd.Timedelta(hours=hours)
        return (
            (history_df["published_at_local"] >= start_time) &
            (history_df["published_at_local"] < candidate_time)
        ).sum()

    eligible_history = format_history[
        format_history["published_at_local"] + pd.Timedelta(days=h_days) <= candidate_time
    ].copy()

    recent_history = eligible_history.tail(window_size)
    hist_n_valid_format = len(recent_history)

    if hist_n_valid_format > 0:
        hist_median_log_views_7d_format = recent_history["log_views_7d_est"].median()
        hist_mean_log_views_7d_format = recent_history["log_views_7d_est"].mean()
        hist_std_log_views_7d_format = recent_history["log_views_7d_est"].std()

        valid_targets = (
            recent_history[target_col].dropna()
            if target_col in recent_history.columns
            else pd.Series(dtype=float)
        )

        hist_median_target_format = valid_targets.median() if len(valid_targets) > 0 else np.nan
        hist_mean_target_format = valid_targets.mean() if len(valid_targets) > 0 else np.nan

        baseline_views_7d_candidate = recent_history["views_7d_est"].median()
    else:
        hist_median_log_views_7d_format = np.nan
        hist_mean_log_views_7d_format = np.nan
        hist_std_log_views_7d_format = np.nan
        hist_median_target_format = np.nan
        hist_mean_target_format = np.nan
        baseline_views_7d_candidate = np.nan

    return {
        "channel_posts_before": channel_posts_before,
        "channel_format_posts_before": channel_format_posts_before,

        "hours_since_last_post": hours_since_last_post,
        "has_previous_post": has_previous_post,
        "hours_since_last_format_post": hours_since_last_format_post,
        "has_previous_format_post": has_previous_format_post,

        "posts_last_24h": count_posts_last_hours(channel_history, 24),
        "posts_last_48h": count_posts_last_hours(channel_history, 48),
        "posts_last_168h": count_posts_last_hours(channel_history, 168),

        "format_posts_last_24h": count_posts_last_hours(format_history, 24),
        "format_posts_last_48h": count_posts_last_hours(format_history, 48),
        "format_posts_last_168h": count_posts_last_hours(format_history, 168),

        "hist_n_valid_format": hist_n_valid_format,
        "hist_median_log_views_7d_format": hist_median_log_views_7d_format,
        "hist_mean_log_views_7d_format": hist_mean_log_views_7d_format,
        "hist_std_log_views_7d_format": hist_std_log_views_7d_format,
        "hist_median_target_format": hist_median_target_format,
        "hist_mean_target_format": hist_mean_target_format,

        "baseline_views_7d_candidate": baseline_views_7d_candidate
    }


def build_candidate_feature_table(
    df_history,
    base_video_row,
    candidate_times_df,
    all_feature_cols,
    h_days=7,
    window_size=10
):
    rows = []

    channel_id = base_video_row["channel_id"]
    is_short = int(base_video_row["is_short"])

    for candidate_time in candidate_times_df["candidate_published_at_local"]:
        candidate_time = pd.Timestamp(candidate_time)

        row = {}

        for col in all_feature_cols:
            row[col] = base_video_row[col] if col in base_video_row.index else np.nan

        publish_hour = candidate_time.hour
        publish_day_num = candidate_time.dayofweek

        row["publish_hour_sin"] = np.sin(2 * np.pi * publish_hour / 24)
        row["publish_hour_cos"] = np.cos(2 * np.pi * publish_hour / 24)

        row["publish_day_sin"] = np.sin(2 * np.pi * publish_day_num / 7)
        row["publish_day_cos"] = np.cos(2 * np.pi * publish_day_num / 7)

        row["is_weekend"] = int(publish_day_num in [5, 6])

        history_features = compute_history_for_candidate(
            df_history=df_history,
            channel_id=channel_id,
            is_short=is_short,
            candidate_time=candidate_time,
            h_days=h_days,
            window_size=window_size
        )

        row.update(history_features)

        row["candidate_published_at_local"] = candidate_time
        row["candidate_dayofweek"] = DAY_NAMES_ES[publish_day_num]
        row["candidate_hour"] = publish_hour
        row["channel_title"] = base_video_row["channel_title"]
        row["is_short"] = is_short

        rows.append(row)

    return pd.DataFrame(rows)


def impute_candidate_features(candidate_df, reference_df):
    candidate_df = candidate_df.copy()

    for col in all_feature_cols:
        if col not in candidate_df.columns:
            candidate_df[col] = np.nan

    numeric_features = [
        col for col in all_feature_cols
        if col not in categorical_features
    ]

    for col in numeric_features:
        if candidate_df[col].isna().any():
            fill_value = reference_df[col].median() if col in reference_df.columns else 0
            if pd.isna(fill_value):
                fill_value = 0
            candidate_df[col] = candidate_df[col].fillna(fill_value)

    for col in categorical_features:
        if candidate_df[col].isna().any():
            if col in reference_df.columns:
                mode_values = reference_df[col].dropna().astype(str).mode()
                fill_value = mode_values.iloc[0] if len(mode_values) > 0 else "Unknown"
            else:
                fill_value = "Unknown"

            candidate_df[col] = candidate_df[col].fillna(fill_value)

        candidate_df[col] = candidate_df[col].astype(str)

    return candidate_df


def recommend_publication_times(
    model,
    df_history,
    base_video_row,
    start_datetime,
    days_ahead=7,
    top_k=3,
    h_days=7,
    window_size=10
):
    candidate_times_df = generate_candidate_slots(
        start_datetime=start_datetime,
        days_ahead=days_ahead
    )

    candidate_df = build_candidate_feature_table(
        df_history=df_history,
        base_video_row=base_video_row,
        candidate_times_df=candidate_times_df,
        all_feature_cols=all_feature_cols,
        h_days=h_days,
        window_size=window_size
    )

    candidate_df = impute_candidate_features(candidate_df, df_history)

    candidate_df["predicted_target_relative_log_views_7d"] = model.predict(
        candidate_df[all_feature_cols]
    )

    candidate_df["predicted_ratio_vs_baseline"] = np.exp(
        candidate_df["predicted_target_relative_log_views_7d"]
    )

    candidate_df["predicted_pct_vs_baseline"] = (
        (candidate_df["predicted_ratio_vs_baseline"] - 1) * 100
    )

    candidate_df["predicted_views_7d"] = (
        (1 + candidate_df["baseline_views_7d_candidate"])
        * candidate_df["predicted_ratio_vs_baseline"]
        - 1
    )

    candidate_df = candidate_df.sort_values(
        "predicted_target_relative_log_views_7d",
        ascending=False
    )

    return candidate_df.head(top_k), candidate_df


# ============================================================
# Interfaz
# ============================================================

st.title("📈 Recomendador de horarios de publicación en YouTube")

st.markdown(
    """
    Esta aplicación recomienda los mejores momentos de publicación usando un modelo de machine learning
    entrenado sobre datos históricos de vídeos de YouTube.
    """
)

tab_resumen, tab_recomendador = st.tabs(
    ["📊 Resumen", "🕒 Recomendador"]
)


# ============================================================
# Resumen
# ============================================================

with tab_resumen:
    st.header("Resumen del dataset")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Vídeos", f"{len(df_fe):,}")
    col2.metric("Canales", f"{df_fe['channel_id'].nunique():,}")

    if target_col in df_fe.columns:
        col3.metric("Vídeos con target", f"{df_fe[target_col].notna().mean() * 100:.1f}%")
    else:
        col3.metric("Vídeos con target", "N/A")

    col4.metric("Features del modelo", len(all_feature_cols))

    st.write("Rango temporal:")
    st.write(f"{df_fe['published_at_local'].min()} → {df_fe['published_at_local'].max()}")

    st.subheader("Canales con más vídeos")

    top_channels = (
        df_fe.groupby("channel_title")
        .agg(
            n_videos=("video_id", "count"),
            median_views=("view_count", "median")
        )
        .sort_values("n_videos", ascending=False)
        .head(20)
        .reset_index()
    )

    st.dataframe(top_channels, use_container_width=True)


# ============================================================
# Recomendador
# ============================================================

with tab_recomendador:
    st.header("Recomendador top-k")

    st.markdown(
        """
        Puedes generar recomendaciones usando un vídeo existente del dataset
        o introduciendo manualmente las características de un nuevo vídeo.
        """
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        input_mode = st.radio(
            "Modo de entrada",
            ["Vídeo existente del dataset", "Introducir vídeo manualmente"],
            horizontal=False
        )

        channels = sorted(df_fe["channel_title"].dropna().unique())

        selected_channel = st.selectbox("Canal", channels)

        format_label = st.radio(
            "Formato",
            ["No Short", "Short"],
            horizontal=True
        )

        selected_is_short = 1 if format_label == "Short" else 0

        channel_all_df = df_fe[
            df_fe["channel_title"] == selected_channel
        ].copy()

        channel_format_df = df_fe[
            (df_fe["channel_title"] == selected_channel) &
            (df_fe["is_short"] == selected_is_short)
        ].copy()

        if len(channel_all_df) == 0:
            st.warning("No hay histórico para este canal.")
            st.stop()

        # ----------------------------------------------------
        # Modo 1: vídeo existente
        # ----------------------------------------------------

        if input_mode == "Vídeo existente del dataset":

            if len(channel_format_df) == 0:
                st.warning("No hay vídeos de ese canal/formato en el dataset.")
                st.stop()

            channel_format_df = channel_format_df.sort_values(
                "published_at_local",
                ascending=False
            )

            channel_format_df["video_label"] = (
                channel_format_df["published_at_local"].astype(str)
                + " | "
                + channel_format_df["title"].astype(str)
            )

            selected_video_label = st.selectbox(
                "Vídeo de referencia",
                channel_format_df["video_label"].head(30).tolist()
            )

            base_video_row = channel_format_df[
                channel_format_df["video_label"] == selected_video_label
            ].iloc[0]

        # ----------------------------------------------------
        # Modo 2: vídeo manual
        # ----------------------------------------------------

        else:
            st.subheader("Características del nuevo vídeo")

            title_input = st.text_input("Título del vídeo", value="Nuevo vídeo")

            description_input = st.text_area(
                "Descripción",
                value="",
                height=100
            )

            tags_input = st.text_input(
                "Tags",
                value="",
                help="Puedes separarlos por comas. Ejemplo: fútbol, reto, vlog"
            )

            duration_minutes_input = st.number_input(
                "Duración del vídeo en minutos",
                min_value=0.1,
                max_value=600.0,
                value=1.0 if selected_is_short == 1 else 20.0,
                step=0.5
            )

            category_options = sorted(
                df_fe["category_id"]
                .fillna("Unknown")
                .astype(str)
                .unique()
            )

            if len(category_options) == 0:
                category_options = ["Unknown"]

            default_category = (
                channel_all_df["category_id"]
                .fillna("Unknown")
                .astype(str)
                .mode()
            )

            default_category_value = (
                default_category.iloc[0]
                if len(default_category) > 0
                else category_options[0]
            )

            default_category_index = (
                category_options.index(default_category_value)
                if default_category_value in category_options
                else 0
            )

            category_input = st.selectbox(
                "Categoría YouTube",
                category_options,
                index=default_category_index
            )

            country_options = sorted(
                df_fe["channel_country"]
                .fillna("Unknown")
                .astype(str)
                .unique()
            )

            if len(country_options) == 0:
                country_options = ["Unknown"]

            default_country = (
                channel_all_df["channel_country"]
                .fillna("Unknown")
                .astype(str)
                .mode()
            )

            default_country_value = (
                default_country.iloc[0]
                if len(default_country) > 0
                else "Unknown"
            )

            default_country_index = (
                country_options.index(default_country_value)
                if default_country_value in country_options
                else 0
            )

            country_input = st.selectbox(
                "País del canal",
                country_options,
                index=default_country_index
            )

            definition_options = sorted(
                df_fe["definition"]
                .fillna("Unknown")
                .astype(str)
                .unique()
            )

            if len(definition_options) == 0:
                definition_options = ["hd", "sd", "Unknown"]

            default_definition = (
                channel_all_df["definition"]
                .fillna("Unknown")
                .astype(str)
                .mode()
            )

            default_definition_value = (
                default_definition.iloc[0]
                if len(default_definition) > 0
                else definition_options[0]
            )

            default_definition_index = (
                definition_options.index(default_definition_value)
                if default_definition_value in definition_options
                else 0
            )

            definition_input = st.selectbox(
                "Definición",
                definition_options,
                index=default_definition_index
            )

            caption_input = st.checkbox(
                "Tiene captions/subtítulos",
                value=True
            )

            base_video_row = build_manual_base_video_row(
                df_history=df_fe,
                channel_title=selected_channel,
                is_short=selected_is_short,
                title=title_input,
                description=description_input,
                tags_text=tags_input,
                duration_minutes=duration_minutes_input,
                category_id=category_input,
                channel_country=country_input,
                definition=definition_input,
                caption=int(caption_input)
            )

        # ----------------------------------------------------
        # Parámetros comunes
        # ----------------------------------------------------

        days_ahead = st.slider("Días a futuro", 3, 14, 7)
        top_k = st.slider("Número de recomendaciones", 3, 10, 3)
        min_delay_hours = st.slider("Antelación mínima en horas", 0, 48, 3)

        start_datetime = (
            pd.Timestamp.now(tz="Europe/Madrid")
            .ceil("h")
            + pd.Timedelta(hours=min_delay_hours)
        )

        run_button = st.button("Generar recomendación", type="primary")

    with col_right:
        st.subheader("Vídeo usado para la recomendación")

        st.write(f"**Modo:** {input_mode}")
        st.write(f"**Canal:** {base_video_row['channel_title']}")
        st.write(f"**Formato:** {'Short' if int(base_video_row['is_short']) == 1 else 'No Short'}")

        if input_mode == "Vídeo existente del dataset":
            st.write(f"**Título:** {base_video_row['title']}")
            st.write(f"**Fecha original:** {base_video_row['published_at_local']}")



        else:
            st.write(f"**Título introducido:** {title_input}")
            st.write(f"**Duración:** {duration_minutes_input:.1f} minutos")
            st.write(f"**Categoría:** {category_input}")
            st.write(f"**País:** {country_input}")
            st.write(f"**Definición:** {definition_input}")
            st.write(f"**Tags estimados:** {count_tags_from_text(tags_input)}")

        st.info(
            """
            El sistema no usa visualizaciones, likes ni comentarios del vídeo nuevo.
            Solo utiliza variables conocidas antes de publicar y el histórico del canal.
            """
        )

    if run_button:
        top_recs, all_recs = recommend_publication_times(
            model=model,
            df_history=df_fe,
            base_video_row=base_video_row,
            start_datetime=start_datetime,
            days_ahead=days_ahead,
            top_k=top_k,
            h_days=7,
            window_size=10
        )

        st.session_state["top_recs"] = top_recs
        st.session_state["all_recs"] = all_recs

    if "top_recs" in st.session_state:
        st.subheader("Top recomendaciones")

        top_recs = st.session_state["top_recs"]

        display_cols = [
            "candidate_published_at_local",
            "candidate_dayofweek",
            "candidate_hour",
            "baseline_views_7d_candidate",
            "predicted_target_relative_log_views_7d",
            "predicted_ratio_vs_baseline",
            "predicted_pct_vs_baseline",
            "predicted_views_7d",
            "hours_since_last_post",
            "posts_last_24h",
            "posts_last_168h",
            "hist_n_valid_format"
        ]

        st.dataframe(
            top_recs[display_cols].style.format({
                "baseline_views_7d_candidate": "{:,.0f}",
                "predicted_target_relative_log_views_7d": "{:.3f}",
                "predicted_ratio_vs_baseline": "{:.2f}",
                "predicted_pct_vs_baseline": "{:.1f}%",
                "predicted_views_7d": "{:,.0f}",
                "hours_since_last_post": "{:.1f}"
            }),
            use_container_width=True
        )

        best = top_recs.iloc[0]

        st.success(
            f"Mejor horario recomendado: **{best['candidate_dayofweek']} "
            f"a las {int(best['candidate_hour'])}:00**. "
            f"Ratio esperado frente al baseline: **{best['predicted_ratio_vs_baseline']:.2f}x**."
        )


