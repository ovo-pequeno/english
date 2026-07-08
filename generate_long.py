# =========================================================
# ひとくち英語｜長尺（やさしい英語の短編ストーリー聞き流し・複数話）
# 【完全創作】Geminiが学習者向けのやさしい英語を作成。
# 現地語読み上げ（gTTS en）＋現地語と和訳を表示。15分以内に収める。
# Gemini → gTTS(en) → MoviePy → YouTube API / 横型1920x1080
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
# 電話番号認証なしでも上げられるよう15分以内に収める。
# 4話×(タイトル+10〜14文)＋前後カードで概ね12〜14分想定。
NUM_STORIES  = int(os.environ.get("NUM_STORIES", "4"))
MAX_SECONDS  = 14 * 60      # これを超えたら以降の話を打ち切って15分未満を厳守
OUT_DIR  = "out_l"
TMP_DIR  = "tmp_l"
LOG_PATH = "used_log_long.json"
AVOID_RECENT = 40

BG_IMAGE = "assets/bg_long.png" if os.path.exists("assets/bg_long.png") else None
BGM_PATH = "assets/bgm.mp3" if os.path.exists("assets/bgm.mp3") else None
BGM_VOLUME = 0.10

client = genai.Client(api_key=GEMINI_API_KEY)

W, H = 1920, 1080
FPS = 10

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_TGT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
if not os.path.exists(FONT_TGT):
    FONT_TGT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

BG_COLOR      = (18, 24, 40)
TGT_COLOR     = "white"
JA_COLOR      = "#9FD0FF"
STROKE_COLOR  = "#000000"
TGT_FONTSIZE  = 64
JA_FONTSIZE   = 44
HEADER_FONTSIZE = 34
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
Write a short, easy-to-listen original story in English (level: {LEVEL}).

Rules:
- Completely original. No real people, brands, or specific place names.
- Use simple, natural, everyday English suitable for learners.
- About 10-14 short sentences (a small complete story).
- Provide an accurate Japanese translation for each sentence.

Output JSON only (no markdown, no extra text):
{{
  "title_tgt": "short title in English",
  "title_ja": "日本語タイトル（20字以内）",
  "summary": "one-line Japanese summary for dedup log (40 chars or less)",
  "sentences": [
    {{"tgt": "one sentence in English.", "ja": "その日本語訳"}}
  ]
}}
sentences: 10-14 items.{avoid_text}
"""
    cfg = genai_types.GenerateContentConfig(max_output_tokens=4096, temperature=1.05) if genai_types else None
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
        size = (W - 260, None)
    # 塗りと縁取りを1回で描く（2枚重ねだと折り返しズレで影がずれるため単一クリップに）
    clip = TextClip(text, color=color, stroke_color=STROKE_COLOR, stroke_width=stroke_w,
                    font=font, fontsize=fontsize, method="caption",
                    size=size, align="center", interline=14).set_duration(duration)
    return clip.set_position(("center", ypos))


def make_scene(tgt_text, ja_text, audio_file, header_label):
    narration = AudioFileClip(audio_file)
    duration = narration.duration + 0.6
    layers = [make_bg(duration)]
    layers.append(make_outlined(header_label, duration, FONT, HEADER_FONTSIZE,
                                "#7FB0E0", stroke_w=4, ypos=int(H * 0.08)))
    layers.append(make_outlined(tgt_text, duration, FONT_TGT, TGT_FONTSIZE, TGT_COLOR,
                                stroke_w=6, ypos=int(H * 0.40)))
    if ja_text:
        layers.append(make_outlined(ja_text, duration, FONT, JA_FONTSIZE, JA_COLOR,
                                    stroke_w=5, ypos=int(H * 0.66)))
    scene = CompositeVideoClip(layers, size=(W, H)).set_duration(duration)
    if duration > narration.duration + 0.02:
        narration = CompositeAudioClip([narration]).set_duration(duration)
    return scene.set_audio(narration)


def make_card(text, audio_file):
    narration = AudioFileClip(audio_file)
    duration = narration.duration + 0.8
    layers = [ColorClip(size=(W, H), color=(12, 16, 28), duration=duration)]
    layers.append(make_outlined(text, duration, FONT_TGT, 78, "white", stroke_w=6, ypos="center"))
    scene = CompositeVideoClip(layers, size=(W, H)).set_duration(duration)
    if duration > narration.duration + 0.02:
        narration = CompositeAudioClip([narration]).set_duration(duration)
    return scene.set_audio(narration)


def render(scene, out_path):
    scene.write_videofile(out_path, fps=FPS, codec="libx264",
                          audio_codec="aac", preset="ultrafast", logger=None)
    try:
        if scene.audio is not None:
            scene.audio.close()
    except Exception:
        pass
    scene.close(); del scene; gc.collect()


def _dur(path):
    c = AudioFileClip(path); d = c.duration; c.close(); return d


def build_video(stories):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    main_title = stories[0].get("title_ja", "ひとくち英語")
    safe = main_title
    for ch in r'\/:*?"<>|':
        safe = safe.replace(ch, "")
    output_path = os.path.join(OUT_DIR, f"{safe.strip()[:60]}.mp4")

    clip_paths = []
    idx = 0
    total = 0.0
    used_stories = 0
    for si, story in enumerate(stories):
        label = f"Story {si+1}  {story.get('title_tgt','')}"
        a = make_audio_tgt(story.get("title_tgt", f"Story {si+1}"), f"a_{idx}.mp3")
        card_d = _dur(a) + 0.8
        # 15分制限：このカードを足すと超えるなら打ち切り
        if total + card_d > MAX_SECONDS and used_stories > 0:
            os.remove(a); break
        p = f"{TMP_DIR}/clip_{idx:04d}.mp4"
        render(make_card(story.get("title_tgt", f"Story {si+1}"), a), p)
        clip_paths.append(p); os.remove(a); idx += 1; total += card_d
        stopped = False
        for s in story["sentences"]:
            tgt = (s.get("tgt") or "").strip()
            ja = (s.get("ja") or "").strip()
            if not tgt:
                continue
            a = make_audio_tgt(tgt, f"a_{idx}.mp3")
            sc_d = _dur(a) + 0.6
            if total + sc_d > MAX_SECONDS:
                os.remove(a); stopped = True; break
            print(f"  [Story{si+1}] {tgt[:34]}...")
            p = f"{TMP_DIR}/clip_{idx:04d}.mp4"
            render(make_scene(tgt, ja, a, label), p)
            clip_paths.append(p); os.remove(a); idx += 1; total += sc_d
        used_stories += 1
        if stopped:
            break

    # エンディング
    end_txt = "Thank you for listening!"
    a = make_audio_tgt("Thank you for listening!", f"a_{idx}.mp3")
    p = f"{TMP_DIR}/clip_{idx:04d}.mp4"
    render(make_card(end_txt, a), p)
    clip_paths.append(p); os.remove(a); idx += 1

    print(f"  connect {len(clip_paths)} scenes (~{int(total)}s)...")
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
    return output_path, main_title, used_stories


def get_youtube():
    creds = Credentials(token=None, refresh_token=YT_REFRESH_TOKEN,
                        client_id=YT_CLIENT_ID, client_secret=YT_CLIENT_SECRET,
                        token_uri="https://oauth2.googleapis.com/token")
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def upload(youtube, path, title, stories):
    titles = " / ".join(s.get("title_tgt", "") for s in stories)
    description = (
        "やさしい英語の短いストーリーを集めた、聞き流し英語リスニング（現地語＋和訳つき）。\n"
        "作業用・睡眠用・通勤用にどうぞ。\n"
        f"Stories: {titles}\n\n#英語 #英語リスニング #英語学習 #聞き流し #作業用BGM"
    )
    body = {
        "snippet": {
            "title": title[:100],
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
    stories = []
    for i in range(NUM_STORIES):
        print(f"Story {i+1}/{NUM_STORIES} 生成中...")
        st = generate_story(avoid + [s.get("summary", "") for s in stories])
        stories.append(st)
        print(f"  {st.get('title_tgt')} ({len(st.get('sentences', []))} sent)")
        time.sleep(2)

    path, title, used = build_video(stories)
    used_stories = stories[:used] if used else stories
    print(f"done: {path} (使用 {used}話)")
    youtube = get_youtube()
    res = upload(youtube, path, title, used_stories)
    print(f"uploaded: https://www.youtube.com/watch?v={res['id']}")
    for st in used_stories:
        log.append({"title": st.get("title_ja", ""), "summary": st.get("summary", "")})
    save_log(log)
    print(f"log: {len(log)} items")


if __name__ == "__main__":
    main()
