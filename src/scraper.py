import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_youtube_data(url: str, api_key: str, max_comments: int = 200, sort_by: str = "relevance") -> tuple[dict, list[dict]]:
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("URL YouTube tidak valid. Pastikan tautan mengarah ke video YouTube.")

    youtube = build("youtube", "v3", developerKey=api_key)

    try:
        # 1. Ambil metadata video (Judul, Channel, Views) -> Biaya: 1 unit kuota
        video_resp = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        ).execute()

        if not video_resp.get("items"):
            raise ValueError("Video tidak ditemukan atau bersifat pribadi (private).")

        video_item = video_resp["items"][0]
        metadata = {
            "title": video_item["snippet"]["title"],
            "channel": video_item["snippet"]["channelTitle"],
            "views": int(video_item["statistics"].get("viewCount", 0))
        }

        # 2. Ambil komentar secara bertahap (maksimal 100 per halaman) -> Biaya: 1 unit per 100 komentar
        comments = []
        next_page_token = None
        order_param = "relevance" if sort_by == "popular" else "time"

        while len(comments) < max_comments:
            batch_size = min(100, max_comments - len(comments))
            comment_resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=batch_size,
                order=order_param,
                textFormat="plainText",
                pageToken=next_page_token
            ).execute()

            for item in comment_resp.get("items", []):
                top_comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "author": top_comment.get("authorDisplayName", "Anonim"),
                    "text": top_comment.get("textDisplay", ""),
                    "likes": int(top_comment.get("likeCount", 0)),
                    "time": top_comment.get("publishedAt", "")[:10]
                })

            next_page_token = comment_resp.get("nextPageToken")
            if not next_page_token:
                break

        return metadata, comments

    except HttpError as e:
        if e.resp.status == 400:
            raise RuntimeError("API Key YouTube tidak valid. Periksa kembali konfigurasi secrets.")
        elif e.resp.status == 403:
            raise RuntimeError("Akses ditolak: Komentar pada video ini dinonaktifkan atau kuota API harian habis.")
        elif e.resp.status == 404:
            raise RuntimeError("Video tidak ditemukan atau telah dihapus.")
        raise RuntimeError(f"Gangguan layanan YouTube API (Kode {e.resp.status}).")
    except Exception:
        raise RuntimeError("Gagal terhubung ke server YouTube. Periksa koneksi internet Anda.")