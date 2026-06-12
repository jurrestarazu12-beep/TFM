import os
import pandas as pd
from datetime import datetime, timezone

from youtube_client import (
    get_youtube_client,
    search_channels,
    get_channel_info,
    get_uploads_playlist_id,
    get_video_ids_from_playlist_by_age,
    get_video_details,
    parse_youtube_datetime,
    parse_iso8601_duration
)


# ============================================================
# 1. CANALES A BUSCAR
# ==========================================================
'''
CHANNEL_QUERIES = [
"Mikecrack", "elrubiusOMG", "VEGETTA777", "invictor", "Chuster", "AuronPlay", "Las Ratitas", "FC Barcelona", "ElTrollino", "Nahz", "Real Madrid CF", "Albertisment", 
"TheWillyrex", "TheGrefg", "Willyrex", "Luli Pampín", "iTownGamePlay", "Makiman131", "Adexe & Nau", "GymVirtual", "Ben Elen", "Timba Vk", "RaptorGamer", "LuzuGames", 
"DaniRep Happy", "ExpCaseros", "ZarcortGame", "BersGamer", "Pica-Pica", "Dalas Review", "DoctorePoLLo", "DJMaRiiO", "Jordi Wild", "The Wild Project", "Ibai", "wismichu", 
"byViruZz", "Paula Gonu", "Nil Ojeda", "Marta Diaz", "Ampeterby7", "Vicens", "Agustin51", "Tarifa", "Logan G", "Shoot_iN", "Salva", "NexxuzHD", "sTaXxCraft", "bysTaXx", "Alexby11", 
"Mangelrogel", "SrCheeto", "LMDShow", "Spursito", "Xbuyer", "MiniBuyer", "RobertPG", "Kolderiu", "Cacho01", "MiiKeLMsT", "Aisack",
]
'''
'''
CHANNEL_QUERIES = [ "Austin Evans",
  "Mrwhosetheboss",
  "Gamers Nexus",
  "IGN",
  "GameSpot",
  "Markiplier",
  "jacksepticeye",
  "PewDiePie",
  "Dream",
  "TommyInnit",
  "LazarBeam",
  "Sidemen",
  "KSI",
  "Yes Theory",
  "Casey Neistat",
]

'''

CHANNEL_QUERIES = [
  "Nate Gentile",
  "Topes de Gama",
  "ProAndroid",
  "Clipset",
  "Urban Tecno",
  "Tecnonauta",
  "Xataka TV",
  "Suprapixel",
  "Just Unboxing",
  "NewEsc",
  "El Output",
  "Isa Marcial",
  "Carlos Vassan",
  "TecnoLike Plus",
  "SupraPixel",
  "The Verge",
  "CNET",
  "TechLinked",
  "ShortCircuit",
  "Dave2D",
  "iJustine",
  "UrAvgConsumer",
  "JerryRigEverything",
  "Android Authority",
  "MrMobile",
  "Hardware Canucks",
  "JayzTwoCents",
  "Paul's Hardware",
  "Hardware Unboxed",
  "Optimum",
  "TechSource",
  "Techquickie",
  "TechAltar",
  "Techmoan",
  "ExplainingComputers",
  "Hablemos de videojuegos",

]
 



"Mikecrack", "elrubiusOMG", "VEGETTA777", "invictor", "Chuster", "AuronPlay", "Las Ratitas", "FC Barcelona", "ElTrollino", "Nahz", "Real Madrid CF", "Albertisment", "TheWillyrex", "TheGrefg", "Willyrex", "Luli Pampín", "iTownGamePlay", "Makiman131", "Adexe & Nau", "GymVirtual", "Ben Elen", "Timba Vk", "RaptorGamer", "LuzuGames", "DaniRep Happy", "ExpCaseros", "ZarcortGame", "BersGamer", "Pica-Pica", "Dalas Review", "DoctorePoLLo", "DJMaRiiO", "Jordi Wild", "The Wild Project", "Ibai", "wismichu", "byViruZz", "Paula Gonu", "Nil Ojeda", "Marta Diaz", "Ampeterby7", "Vicens", "Agustin51", "Tarifa", "Logan G", "Shoot_iN", "Salva", "NexxuzHD", "sTaXxCraft", "bysTaXx", "Alexby11", "Mangelrogel", "SrCheeto", "LMDShow", "Spursito", "Xbuyer", "MiniBuyer", "RobertPG", "Kolderiu", "Cacho01", "MiiKeLMsT", "Aisack",
# ============================================================
# 2. CONFIGURACIÓN
# ============================================================

SHORT_MAX_DURATION_SECONDS = 180

# Ventana de vídeos recientes.
# Puedes ajustarla, pero de momento mantenemos lo que ya tenías.
MIN_VIDEO_AGE_DAYS = 6
MAX_VIDEO_AGE_DAYS = 30

MAX_VIDEOS_TO_SCAN_PER_CHANNEL = 500

DEBUG_DURATIONS = False

RAW_DIR = "Data/raw"
MASTER_DATASET_PATH = os.path.join(RAW_DIR, "youtube_videos_master.csv")


# ============================================================
# 3. FUNCIONES AUXILIARES
# ============================================================

def calculate_video_age_days(published_at):
    published_datetime = parse_youtube_datetime(published_at)

    if published_datetime is None:
        return None

    now = datetime.now(timezone.utc)

    return (now - published_datetime).days


def is_short_video(duration_seconds):
    if duration_seconds is None:
        return None

    return duration_seconds <= SHORT_MAX_DURATION_SECONDS


def load_existing_video_ids(master_path):
    if not os.path.exists(master_path):
        return set()

    if os.path.getsize(master_path) == 0:
        return set()

    df_existing = pd.read_csv(master_path, usecols=["video_id"])

    existing_video_ids = (
        df_existing["video_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return set(existing_video_ids)


def update_master_dataset(new_df, master_path):
    if os.path.exists(master_path) and os.path.getsize(master_path) > 0:
        old_df = pd.read_csv(master_path)
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined_df = new_df.copy()

    if len(combined_df) == 0:
        combined_df.to_csv(
            master_path,
            index=False,
            encoding="utf-8-sig"
        )
        return combined_df, 0

    rows_before = len(combined_df)

    combined_df = combined_df.drop_duplicates(
        subset=["video_id"],
        keep="first"
    )

    rows_after = len(combined_df)

    duplicates_removed = rows_before - rows_after

    combined_df.to_csv(
        master_path,
        index=False,
        encoding="utf-8-sig"
    )

    return combined_df, duplicates_removed


# ============================================================
# 4. CREACIÓN DEL DATASET DE ESTA EJECUCIÓN
# ============================================================

def build_raw_dataset(
    channel_queries,
    existing_video_ids,
    min_video_age_days=0,
    max_video_age_days=45,
    max_videos_to_scan_per_channel=500
):
    youtube = get_youtube_client()

    all_rows = []
    collected_video_ids_this_run = set()

    for query in channel_queries:
        print(f"\nBuscando canal: {query}")

        channels = search_channels(
            youtube=youtube,
            query=query,
            max_results=1
        )

        if not channels:
            print(f"No se encontró canal para: {query}")
            continue

        channel_id = channels[0]["channel_id"]

        channel_info = get_channel_info(
            youtube=youtube,
            channel_id=channel_id
        )

        if channel_info is None:
            print(f"No se pudo obtener información del canal: {query}")
            continue

        print(f"Canal encontrado: {channel_info['channel_title']}")

        uploads_playlist_id = get_uploads_playlist_id(
            youtube=youtube,
            channel_id=channel_id
        )

        if uploads_playlist_id is None:
            print(f"No se encontró uploads playlist para: {query}")
            continue

        candidate_video_ids = get_video_ids_from_playlist_by_age(
            youtube=youtube,
            playlist_id=uploads_playlist_id,
            min_age_days=min_video_age_days,
            max_age_days=max_video_age_days,
            max_videos_to_scan=max_videos_to_scan_per_channel
        )

        print(
            f"Vídeos candidatos entre {min_video_age_days} "
            f"y {max_video_age_days} días: {len(candidate_video_ids)}"
        )

        if not candidate_video_ids:
            continue

        new_video_ids = []

        for video_id in candidate_video_ids:
            video_id = str(video_id).strip()

            if video_id in existing_video_ids:
                continue

            if video_id in collected_video_ids_this_run:
                continue

            new_video_ids.append(video_id)

        print(f"Vídeos ya existentes en master descartados: {len(candidate_video_ids) - len(new_video_ids)}")
        print(f"Vídeos nuevos para recuperar: {len(new_video_ids)}")

        if not new_video_ids:
            continue

        videos = get_video_details(
            youtube=youtube,
            video_ids=new_video_ids
        )

        print(f"Vídeos recuperados desde API: {len(videos)}")

        videos_added_for_channel = 0
        shorts_count = 0
        non_shorts_count = 0
        unknown_duration_count = 0

        for video in videos:
            video_id = str(video["video_id"]).strip()

            if video_id in existing_video_ids:
                continue

            if video_id in collected_video_ids_this_run:
                continue

            video_age_days = calculate_video_age_days(video["published_at"])

            if video_age_days is None:
                continue

            if not (min_video_age_days <= video_age_days <= max_video_age_days):
                continue

            duration_seconds = parse_iso8601_duration(video["duration"])
            is_short = is_short_video(duration_seconds)

            if is_short is True:
                shorts_count += 1
            elif is_short is False:
                non_shorts_count += 1
            else:
                unknown_duration_count += 1

            if DEBUG_DURATIONS:
                print(
                    f"Duración: {str(duration_seconds):>4}s | "
                    f"is_short: {is_short} | "
                    f"Título: {video['title']}"
                )

            row = {
                **video,

                "video_age_days": video_age_days,
                "duration_seconds": duration_seconds,
                "is_short": is_short,

                "collection_min_age_days": min_video_age_days,
                "collection_max_age_days": max_video_age_days,

                "channel_subscriber_count": channel_info["subscriber_count"],
                "channel_total_view_count": channel_info["view_count"],
                "channel_video_count": channel_info["video_count"],
                "channel_published_at": channel_info["published_at"],
                "channel_country": channel_info["country"],

                "query_used": query,
                "data_extracted_at": datetime.now(timezone.utc).isoformat()
            }

            all_rows.append(row)
            collected_video_ids_this_run.add(video_id)
            videos_added_for_channel += 1

        print(f"Shorts añadidos para este canal: {shorts_count}")
        print(f"No shorts añadidos para este canal: {non_shorts_count}")
        print(f"Duración desconocida para este canal: {unknown_duration_count}")
        print(f"Vídeos nuevos añadidos para este canal: {videos_added_for_channel}")

    df = pd.DataFrame(all_rows)

    if len(df) > 0:
        df = df.drop_duplicates(
            subset=["video_id"],
            keep="first"
        )

    return df


# ============================================================
# 5. EJECUCIÓN PRINCIPAL
# ============================================================

def main():

    print("ARCHIVO EJECUTADO:", os.path.abspath(__file__))
    print("CARPETA ACTUAL:", os.getcwd())
    print("RAW_DIR:", os.path.abspath(RAW_DIR))
    print("MASTER_DATASET_PATH:", os.path.abspath(MASTER_DATASET_PATH))
    os.makedirs(RAW_DIR, exist_ok=True)

    existing_video_ids = load_existing_video_ids(MASTER_DATASET_PATH)

    print("\nInicio de creación incremental del dataset")
    print(f"Dataset maestro: {MASTER_DATASET_PATH}")
    print(f"Vídeos ya existentes en master: {len(existing_video_ids)}")

    new_df = build_raw_dataset(
        channel_queries=CHANNEL_QUERIES,
        existing_video_ids=existing_video_ids,
        min_video_age_days=MIN_VIDEO_AGE_DAYS,
        max_video_age_days=MAX_VIDEO_AGE_DAYS,
        max_videos_to_scan_per_channel=MAX_VIDEOS_TO_SCAN_PER_CHANNEL
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_output_path = os.path.join(
        RAW_DIR,
        f"youtube_new_rows_{MIN_VIDEO_AGE_DAYS}_{MAX_VIDEO_AGE_DAYS}_days_{timestamp}.csv"
    )

    if len(new_df) > 0:
        new_df.to_csv(
            run_output_path,
            index=False,
            encoding="utf-8-sig"
        )

    master_df, duplicates_removed = update_master_dataset(
        new_df=new_df,
        master_path=MASTER_DATASET_PATH
    )

    print("\nProceso terminado")
    print(f"Filtro temporal aplicado: vídeos entre {MIN_VIDEO_AGE_DAYS} y {MAX_VIDEO_AGE_DAYS} días")
    print(f"Criterio is_short: duración <= {SHORT_MAX_DURATION_SECONDS} segundos")
    print(f"Vídeos nuevos encontrados en esta ejecución: {len(new_df)}")
    print(f"Duplicados eliminados al actualizar master: {duplicates_removed}")
    print(f"Vídeos únicos acumulados en master: {len(master_df)}")
    print(f"Dataset maestro guardado en: {MASTER_DATASET_PATH}")

    if len(new_df) > 0:
        print(f"CSV de nuevos vídeos de esta ejecución guardado en: {run_output_path}")

        print("\nDistribución de is_short en vídeos nuevos:")
        print(new_df["is_short"].value_counts(dropna=False))

        print("\nResumen de edad de vídeos nuevos:")
        print(new_df["video_age_days"].describe())

        print("\nResumen de duración de vídeos nuevos:")
        print(new_df["duration_seconds"].describe())

        print("\nVídeos nuevos por canal:")
        print(
            new_df
            .groupby("channel_title")["video_id"]
            .nunique()
            .sort_values(ascending=False)
        )

        print("\nColumnas:")
        print(new_df.columns.tolist())
    else:
        print("\nNo se encontraron vídeos nuevos que cumplan el filtro temporal.")


if __name__ == "__main__":
    main()