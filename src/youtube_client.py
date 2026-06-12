import os
import re
from dotenv import load_dotenv
from googleapiclient.discovery import build
from datetime import datetime, timezone


def get_youtube_client():
    """
    Crea y devuelve un cliente de YouTube Data API v3 usando la API key
    almacenada en el archivo .env.
    """
    load_dotenv()

    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise ValueError(
            "No se ha encontrado YOUTUBE_API_KEY. "
            "Comprueba que existe en el archivo .env"
        )

    return build(
        serviceName="youtube",
        version="v3",
        developerKey=api_key
    )


def search_channels(youtube, query, max_results=5):
    """
    Busca canales de YouTube a partir de un texto.

    Parámetros
    ----------
    youtube:
        Cliente de YouTube creado con get_youtube_client().
    query:
        Texto de búsqueda, por ejemplo 'Ibai', 'Dot CSV', 'QuantumFracture'.
    max_results:
        Número máximo de canales a devolver.

    Devuelve
    --------
    list[dict]
        Lista de canales con channel_id, título y descripción.
    """

    request = youtube.search().list(
        part="snippet",
        q=query,
        type="channel",
        maxResults=max_results
    )

    response = request.execute()

    channels = []

    for item in response.get("items", []):
        channel = {
            "channel_id": item["snippet"]["channelId"],
            "channel_title": item["snippet"]["title"],
            "description": item["snippet"].get("description", "")
        }
        channels.append(channel)

    return channels


def get_channel_info(youtube, channel_id):
    """
    Obtiene información básica de un canal concreto.

    Devuelve estadísticas agregadas del canal:
    seguidores, visualizaciones totales, número de vídeos, etc.
    """

    request = youtube.channels().list(
        part="snippet,statistics,contentDetails",
        id=channel_id
    )

    response = request.execute()

    items = response.get("items", [])

    if not items:
        return None

    item = items[0]

    snippet = item["snippet"]
    statistics = item.get("statistics", {})
    content_details = item.get("contentDetails", {})

    return {
        "channel_id": item["id"],
        "channel_title": snippet.get("title"),
        "description": snippet.get("description"),
        "published_at": snippet.get("publishedAt"),
        "country": snippet.get("country"),
        "subscriber_count": int(statistics.get("subscriberCount", 0)),
        "view_count": int(statistics.get("viewCount", 0)),
        "video_count": int(statistics.get("videoCount", 0)),
        "uploads_playlist_id": content_details
            .get("relatedPlaylists", {})
            .get("uploads")
    }


def get_uploads_playlist_id(youtube, channel_id):
    """
    Obtiene el ID de la playlist donde YouTube guarda todos los vídeos
    subidos por un canal.
    """

    channel_info = get_channel_info(youtube, channel_id)

    if channel_info is None:
        return None

    return channel_info["uploads_playlist_id"]


def get_video_ids_from_playlist(youtube, playlist_id, max_results=50):
    """
    Extrae IDs de vídeos desde una playlist.

    Para un canal, usaremos su uploads_playlist_id.
    """

    video_ids = []
    next_page_token = None

    while len(video_ids) < max_results:
        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=min(50, max_results - len(video_ids)),
            pageToken=next_page_token
        )

        response = request.execute()

        for item in response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])

        next_page_token = response.get("nextPageToken")

        if next_page_token is None:
            break

    return video_ids


def get_video_details(youtube, video_ids):
    """
    Obtiene metadatos y estadísticas de una lista de vídeos.

    YouTube permite consultar hasta 50 vídeos por request.
    """

    if not video_ids:
        return []

    videos = []

    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i + 50]

        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch_ids)
        )

        response = request.execute()

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})
            content_details = item.get("contentDetails", {})

            video = {
                "video_id": item.get("id"),
                "channel_id": snippet.get("channelId"),
                "channel_title": snippet.get("channelTitle"),
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "published_at": snippet.get("publishedAt"),
                "category_id": snippet.get("categoryId"),
                "tags": snippet.get("tags", []),
                "view_count": int(statistics.get("viewCount", 0)),
                "like_count": int(statistics.get("likeCount", 0)),
                "comment_count": int(statistics.get("commentCount", 0)),
                "duration": content_details.get("duration"),
                "definition": content_details.get("definition"),
                "caption": content_details.get("caption")
            }

            videos.append(video)

    return videos


def parse_youtube_datetime(date_string):
    """
    Convierte una fecha de YouTube en formato ISO a datetime con zona horaria UTC.

    Ejemplo de entrada:
    '2026-04-24T12:30:00Z'
    """

    if date_string is None:
        return None

    return datetime.fromisoformat(
        date_string.replace("Z", "+00:00")
    )


def get_video_ids_from_playlist_by_age(
    youtube,
    playlist_id,
    min_age_days=30,
    max_age_days=45,
    max_videos_to_scan=500
):
    """
    Extrae IDs de vídeos de una playlist filtrando por edad.

    Solo devuelve vídeos cuya edad esté entre min_age_days y max_age_days.
    Por defecto: entre 30 y 45 días desde la publicación.

    Como la playlist de uploads suele venir ordenada de más reciente a más antigua,
    se detiene cuando encuentra vídeos más antiguos que max_age_days.
    """

    video_ids = []
    next_page_token = None
    scanned_videos = 0

    now = datetime.now(timezone.utc)

    while scanned_videos < max_videos_to_scan:
        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )

        response = request.execute()

        items = response.get("items", [])

        if not items:
            break

        for item in items:
            scanned_videos += 1

            content_details = item.get("contentDetails", {})
            video_id = content_details.get("videoId")

            published_at = content_details.get("videoPublishedAt")

            if published_at is None:
                continue

            published_datetime = parse_youtube_datetime(published_at)

            age_days = (now - published_datetime).days

            if min_age_days <= age_days <= max_age_days:
                video_ids.append(video_id)

            elif age_days > max_age_days:
                return video_ids

            if scanned_videos >= max_videos_to_scan:
                break

        next_page_token = response.get("nextPageToken")

        if next_page_token is None:
            break

    return video_ids


def parse_iso8601_duration(duration):
    """
    Convierte una duración ISO 8601 de YouTube a segundos.

    Ejemplos:
    PT45S -> 45
    PT3M20S -> 200
    PT1H2M10S -> 3730
    """

    if duration is None:
        return None

    pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
    match = re.fullmatch(pattern, duration)

    if not match:
        return None

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    return hours * 3600 + minutes * 60 + seconds