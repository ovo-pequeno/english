# =========================================================
# ひとくち英語｜Shorts（やさしい英語の短編ストーリー聞き流し）
# 【完全創作】Geminiが学習者向けのやさしい英語を作成。
# 現地語を読み上げ（gTTS en）＋画面に現地語と和訳を表示。1分前後。
# Gemini → gTTS(en) → MoviePy → YouTube API / 縦型1080x1920
# =========================================================
import os, re, json, time, gc
from google import genai
try:
    from google.genai import types as genai_types
except Exception:
    genai_types = None
from gtts import gTTS
from pydub import AudioSegment
from moviepy.editor import (
    ColorClip, ImageClip, TextClip, CompositeVideoClip, AudioFileClip, CompositeAudioClip
)
import moviepy.config as cf
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

cf.change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

GEMINI_API_KEY   = os.environ["GEMINI_API_KEY"]
YT_CLIENT_ID     = os.environ["YT_CLIENT_ID"]
YT_CLIENT_SECRET = os.environ["YT_CLIENT_SECRET"]
YT_REFRESH_TOKEN = os.environ["YT_REFRESH_TOKEN"]

PRIVACY = os.environ.get("PRIVACY", "public")
MODEL   = os.environ.get("MODEL", "gemini-2.5-flash")

GTTS_LANG    = "en"
VOICE_SPEED  = 1.0
LEVEL        = os.environ.get("LEVEL", "初級〜中級")
OUT_DIR  = "out_s"
TMP_DIR  = "tmp_s"
LOG_PATH = "used_log_shorts.json"
AVOID_RECENT = 40

BGM_PATH = "assets/bgm.mp3" if os.path.exists("assets/bgm.mp3") else None
BGM_VOLUME = 0.08

client = genai.Client(api_key=GEMINI_API_KEY)

W, H = 1080, 1920
FPS = 10

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_TGT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
if not os.path.exists(FONT_TGT):
    FONT_TGT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

BG_COLOR      = (18, 24, 40)
BG_IMAGE      = "assets/bg_short.png" if os.path.exists("assets/bg_short.png") else None
TGT_COLOR     = "white"
JA_COLOR      = "#9FD0FF"
STROKE_COLOR  = "#000000"
TGT_FONTSIZE  = 70
JA_FONTSIZE   = 48
HEADER_FONTSIZE = 40
HEADER_TEXT   = "One Line a Day"


def load_log():
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_log(log):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)


def generate_story(avoid_summaries, max_retries=5):
    models = [MODEL, "gemini-2.5-flash-lite", "gemini-3.1-flash-lite"]
    avoid_text = ""
    if avoid_summaries:
        joined = "\n".join(f"- {s}" for s in avoid_summaries)
        avoid_text = f"\n\nMake it different from these past stories:\n{joined}"
    prompt = f"""You are a English teacher creating listening material for Japanese learners.
Write a very short, easy-to-listen original story in English (level: {LEVEL}).

Rules:
- Completely original. No real people, brands, or specific place names.
- Use simple, natural, everyday English suitable for learners.
- Keep it short for a 1-minute Shorts video: about 6-9 short sentences.
- Provide an accurate Japanese translation for each sentence.

Output JSON only (no markdown, no extra text):
{{
  "youtube_title": "an appealing Japanese title (25 chars or less)",
  "summary": "one-line summary in Japanese for the dedup log (40 chars or less)",
  "title_tgt": "short title in English",
  "sentences": [
    {{"tgt": "one sentence in English.", "ja": "その日本語訳"}}
  ]
}}
sentences: 6-9 items. Each 'tgt' is one natural English sentence with its 'ja' translation.{avoid_text}
"""
    cfg = genai_types.GenerateContentConfig(temperature=1.05) if genai_types else None
    for attempt in range(max_retries):
        m = models[min(attempt, len(models) - 1)]
        try:
            if cfg:
                resp = client.models.generate_content(model=m, contents=prompt, config=cfg)
            else:
                resp = client.models.generate_content(model=m, contents=prompt)
            text = resp.text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            if not data.get("sentences"):
                raise ValueError("sentencesが空")
            return data
        except Exception as e:
            msg = str(e)
            if ("503" in msg or "429" in msg or "UNAVAILABLE" in msg) and attempt < max_retries - 1:
                time.sleep(20 * (attempt + 1))
            elif attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise


def make_audio_tgt(text, filename):
    if not text.strip():
        AudioSegment.silent(duration=400).export(filename, format="mp3")
        return filename
    tmp = "tmp_" + filename
    gTTS(text=text, lang=GTTS_LANG, slow=False).save(tmp)
    seg = AudioSegment.from_mp3(tmp)
    if VOICE_SPEED and VOICE_SPEED != 1.0:
        seg = seg.speedup(playback_speed=VOICE_SPEED)
    seg = seg + AudioSegment.silent(duration=400)
    seg.export(filename, format="mp3")
    os.remove(tmp)
    return filename


_BG_CACHE = None
def _fit_bg(path):
    global _BG_CACHE
    if _BG_CACHE is None:
        from PIL import Image
        import numpy as np
        resample = getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        _BG_CACHE = np.array(Image.open(path).convert("RGB").resize((W, H), resample))
    return _BG_CACHE


def make_bg(duration):
    if BG_IMAGE and os.path.exists(BG_IMAGE):
        return ImageClip(_fit_bg(BG_IMAGE)).set_duration(duration)
    return ColorClip(size=(W, H), color=BG_COLOR, duration=duration)


def make_outlined(text, duration, font, fontsize, color, stroke_w=6, ypos="center", size=None):
    if size is None:
        size = (W - 120, None)
    common = dict(font=font, fontsize=fontsize, method="caption",
                  size=size, align="center", interline=14)
    stroke = TextClip(text, color=STROKE_COLOR, stroke_color=STROKE_COLOR,
                      stroke_width=stroke_w, **common).set_duration(duration)
    fill = TextClip(text, color=color, **common).set_duration(duration)
    grp = CompositeVideoClip([stroke.set_position("center"), fill.set_position("center")],
                             size=stroke.size).set_duration(duration)
    return grp.set_position(("center", ypos))


def make_scene(tgt_text, ja_text, audio_file):
    narration = AudioFileClip(audio_file)
    duration = narration.duration + 0.6
    layers = [make_bg(duration)]
    layers.append(make_outlined(HEADER_TEXT, duration, FONT_TGT, HEADER_FONTSIZE,
                                 "#7FB0E0", stroke_w=4, ypos=int(H * 0.07)))
    layers.append(make_outlined(tgt_text, duration, FONT_TGT, TGT_FONTSIZE, TGT_COLOR,
                                 stroke_w=7, ypos=int(H * 0.38)))
    if ja_text:
        layers.append(make_outlined(ja_text, duration, FONT, JA_FONTSIZE, JA_COLOR,
                                     stroke_w=5, ypos=int(H * 0.62)))
    scene = CompositeVideoClip(layers, size=(W, H)).set_duration(duration)
    if duration > narration.duration + 0.02:
        narration = CompositeAudioClip([narration]).set_duration(duration)
    return scene.set_audio(narration)


def render_scene(tgt_text, ja_text, audio_file, out_path):
    scene = make_scene(tgt_text, ja_text, audio_file)
    scene.write_videofile(out_path, fps=FPS, codec="libx264",
                          audio_codec="aac", preset="ultrafast", logger=None)
    try:
        if scene.audio is not None:
            scene.audio.close()
    except Exception:
        pass
    scene.close(); del scene; gc.collect()


def build_video(data):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    title = data.get("youtube_title", "ひとくち英語")
    safe = title
    for ch in r'\/:*?"<>|':
        safe = safe.replace(ch, "")
    output_path = os.path.join(OUT_DIR, f"{safe.strip()[:60]}.mp4")

    clip_paths = []
    idx = 0
    for i, s in enumerate(data["sentences"]):
        tgt = (s.get("tgt") or "").strip()
        ja = (s.get("ja") or "").strip()
        if not tgt:
            continue
        print(f"  [{i+1}/{len(data['sentences'])}] {tgt[:30]}...")
        a = make_audio_tgt(tgt, f"a_{idx}.mp3")
        p = f"{TMP_DIR}/clip_{idx:04d}.mp4"
        render_scene(tgt, ja, a, p)
        clip_paths.append(p); os.remove(a); idx += 1

    print(f"  connect {len(clip_paths)} scenes...")
    list_file = f"{TMP_DIR}/list.txt"
    with open(list_file, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{os.path.basename(cp)}'\n")
    master = f"{TMP_DIR}/master.mp4"
    os.system(f'cd {TMP_DIR} && ffmpeg -y -f concat -safe 0 -i list.txt '
              f'-c:v copy -c:a aac master.mp4 -loglevel error')

    if BGM_PATH and os.path.exists(BGM_PATH):
        os.system(
            f'ffmpeg -y -i "{master}" -stream_loop -1 -i "{BGM_PATH}" '
            f'-filter_complex "[1:a]volume={BGM_VOLUME}[b];'
            f'[0:a][b]amix=inputs=2:duration=first:dropout_transition=0[a]" '
            f'-map 0:v -map "[a]" -c:v copy -c:a aac "{output_path}" -loglevel error'
        )
    else:
        os.replace(master, output_path)

    for cp in clip_paths:
        if os.path.exists(cp):
            os.remove(cp)
    for f in [list_file, master]:
        if os.path.exists(f):
            os.remove(f)
    return output_path, title


def get_youtube():
    creds = Credentials(token=None, refresh_token=YT_REFRESH_TOKEN,
                        client_id=YT_CLIENT_ID, client_secret=YT_CLIENT_SECRET,
                        token_uri="https://oauth2.googleapis.com/token")
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def upload(youtube, path, title):
    description = (
        "やさしい英語の短いストーリーで、聞き流し英語リスニング。\n"
        "現地語と和訳つき。作業用・通勤用にどうぞ。\n\n#英語 #英語リスニング #英語学習 #聞き流し #shorts #Shorts"
    )
    body = {
        "snippet": {
            "title": (title + " #shorts")[:100],
            "description": description[:5000],
            "tags": ["英語", "英語リスニング", "英語学習", "聞き流し", "作業用BGM", "リスニング"],
            "categoryId": "27",
            "defaultLanguage": "ja",
        },
        "status": {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(path, chunksize=10 * 1024 * 1024, resumable=True)
    req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None; retry = 0
    while response is None:
        try:
            status, response = req.next_chunk()
            if status:
                print(f"  up {int(status.progress()*100)}%")
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504):
                retry += 1
                if retry > 10: raise
                time.sleep(min(2 ** retry, 60))
            else:
                raise
    return response


def main():
    log = load_log()
    avoid = [e["summary"] for e in log][-AVOID_RECENT:]
    print("生成中...")
    data = generate_story(avoid)
    print(f"  title: {data.get('youtube_title')} ({len(data.get('sentences', []))} sent)")
    path, title = build_video(data)
    print(f"done: {path}")
    youtube = get_youtube()
    res = upload(youtube, path, title)
    print(f"uploaded: https://www.youtube.com/watch?v={res['id']}")
    log.append({"title": data.get("youtube_title", ""), "summary": data.get("summary", "")})
    save_log(log)
    print(f"log: {len(log)} items")


if __name__ == "__main__":
    main()
