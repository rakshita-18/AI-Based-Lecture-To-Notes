# app.py — merged final (includes video->text via yt-dlp + whisper)
import os
import re
import time
import tempfile
import threading
import webbrowser
import shutil

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import firebase_admin
from firebase_admin import credentials, db
from authlib.integrations.flask_client import OAuth

# ML / TTS / YT download
from transformers import pipeline
from gtts import gTTS
import yt_dlp
import whisper

# ---------------- Firebase Config ----------------
FIREBASE_KEY_PATH = "service.json"
FIREBASE_DB_URL = "https://project-3b2b7-default-rtdb.asia-southeast1.firebasedatabase.app"

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})

users_ref = db.reference("/users")  # username/password users

# ---------------- Flask App ----------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")  # use env var in prod

# ---------------- Google OAuth (Authlib) ----------------
app.config["GOOGLE_CLIENT_ID"] = os.getenv("GOOGLE_CLIENT_ID", "YOUR_GOOGLE_CLIENT_ID")
app.config["GOOGLE_CLIENT_SECRET"] = os.getenv("GOOGLE_CLIENT_SECRET", "YOUR_GOOGLE_CLIENT_SECRET")

oauth = OAuth(app)
oauth.register(
    name="google",
    client_id=app.config["GOOGLE_CLIENT_ID"],
    client_secret=app.config["GOOGLE_CLIENT_SECRET"],
    access_token_url="https://oauth2.googleapis.com/token",
    authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
    api_base_url="https://www.googleapis.com/",
    jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
    client_kwargs={"scope": "openid email profile", "prompt": "select_account"},
)

# ---------------- ML Summarizer ----------------
# BART can be heavy; first run downloads weights. Consider caching in prod.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ---------------- Whisper model (for transcription) ----------------
# Use env var WHISPER_MODEL to select model (tiny, base, small, medium, large)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
try:
    whisper_model = whisper.load_model(WHISPER_MODEL)
except Exception as e:
    # If loading fails, set to None and report error at transcription time
    whisper_model = None
    print(f"Warning: whisper model failed to load: {e}")

# ---------------- Helpers ----------------
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

def extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video ID from URL or return ID if valid."""
    if not url_or_id:
        raise ValueError("Empty input")
    url_or_id = url_or_id.strip()
    # direct id pattern
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    # handle common formats: watch?v=, youtu.be/, embed/, shorts/, /v/, /videos/
    patterns = [r"(?:v=|/videos/|embed/|youtu\.be/|/v/|shorts/)([A-Za-z0-9_-]{11})"]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    # try parsing query
    try:
        parsed = urlparse(url_or_id)
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            v = qs["v"][0]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", v):
                return v
    except Exception:
        pass
    if len(url_or_id) >= 11:
        # last-resort heuristic (last 11 chars)
        return url_or_id[-11:]
    raise ValueError("Could not extract YouTube video ID.")

def email_key(email: str) -> str:
    safe = email.lower().strip()
    for ch in [".", "$", "#", "[", "]", "/"]:
        safe = safe.replace(ch, "_")
    return safe

# Simple sentence splitter (avoid heavy deps)
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
def split_sentences(text: str):
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    return parts

def format_transcript(transcript_text: str):
    """
    Build a lightweight formatted output:
      - Headline: first sentence (trimmed)
      - Numbered sections (grouped by N sentences)
      - Bullet points for sentences
    Returns dict with 'html' and 'plain'
    """
    sentences = split_sentences(transcript_text)
    if not sentences:
        return {'html': '<em>No text</em>', 'plain': ''}

    # Headline: first sentence (max ~12 words)
    first_words = ' '.join(sentences[0].split()[:12])
    headline = first_words.strip().rstrip('.!?').title()

    # Group sentences into sections
    sec_size = 5
    sections = [sentences[i:i+sec_size] for i in range(0, len(sentences), sec_size)]

    html_parts = [f'<h2>{headline}</h2>']
    plain_lines = [f'Headline: {headline}', '']

    for idx, sec in enumerate(sections, start=1):
        html_parts.append(f'<h3>{idx}. Section</h3>')
        html_parts.append('<ul>')
        plain_lines.append(f'{idx}.')
        for s in sec:
            bullet = s.strip()
            html_parts.append(f'<li>{bullet}</li>')
            plain_lines.append(f'  - {bullet}')
        html_parts.append('</ul>')
        plain_lines.append('')

    plain = '\n'.join(plain_lines)
    html = '\n'.join(html_parts)
    return {'html': html, 'plain': plain}

def download_audio_from_youtube(url: str, out_dir: str):
    """
    Uses yt-dlp to download best audio and convert to mp3.
    Returns path to created audio file and the yt-dlp info dict.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(out_dir, 'audio.%(ext)s'),
        'quiet': True,
        'noplaylist': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    # find generated file
    for fname in os.listdir(out_dir):
        if fname.startswith('audio') and fname.endswith('.mp3'):
            return os.path.join(out_dir, fname), info
    raise RuntimeError('Downloaded audio file not found')

# ---------------- Pages ----------------
@app.route('/')
def home():
    return render_template("home.html")   # Splash / logo page

@app.route('/login_page')
def login_page():
    return render_template("login.html")

@app.route('/register_page')
def register_page():
    return render_template("register.html")

@app.route('/home1')
def home1():
    if "username" not in session:
        return redirect(url_for("login_page"))
    return render_template("home1.html")

@app.route('/summarize_page')
def summarize_page():
    if "username" not in session:
        return redirect(url_for("login_page"))
    return render_template("summarize.html")

@app.route("/focus")
def focus_page():
    if "username" not in session:
        return redirect(url_for("login_page"))
    return render_template("focus.html")

@app.route("/translate")
def translate_page():
    if "username" not in session:
        return redirect(url_for("login_page"))
    return render_template("translate.html")

@app.route("/recorde")
def recorde_page():
    if "username" not in session:
        return redirect(url_for("login_page"))
    return render_template("recorde.html")

@app.route("/notes")
def notes_page():
    if "username" not in session:
        return redirect(url_for("login_page"))
    return render_template("notes2.html")

# ✅ YouTube Page Loader (GET)
@app.route("/vedio", methods=["GET"])
def vedio_page():
    if "username" not in session:
        return redirect(url_for("login_page"))
    return render_template("vedio.html")

# ---------------- Username/Password Auth ----------------
@app.route('/register', methods=['POST'])
def register():
    username = request.form["username"].strip()
    email = request.form["email"].strip().lower()
    password = request.form["password"]

    users = users_ref.get() or {}
    if username in users:
        flash("Username already exists!", "danger")
        return redirect(url_for("register_page"))

    users_ref.child(username).set({"email": email, "password": password})
    flash("Registration successful! Login now.", "success")
    return redirect(url_for("login_page"))

@app.route('/login', methods=['POST'])
def login():
    username = request.form["username"].strip()
    password = request.form["password"]

    users = users_ref.get() or {}
    if username in users and users[username].get("password") == password:
        session["username"] = username
        session["auth_provider"] = "password"
        flash("Login Successful!", "success")
        return redirect(url_for("home1"))

    flash("Invalid username or password.", "danger")
    return redirect(url_for("login_page"))

@app.route('/logout')
def logout():
    session.pop("username", None)
    session.pop("auth_provider", None)
    flash("Logged out successfully!", "info")
    return redirect(url_for("home"))

# ---------------- Google OAuth Flow ----------------
@app.route("/login/google")
def login_google():
    redirect_uri = url_for("auth_google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route("/auth/google/callback")
def auth_google_callback():
    try:
        token = oauth.google.authorize_access_token()
        userinfo = oauth.google.parse_id_token(token)
    except Exception:
        flash("Google login failed. Please try again.", "danger")
        return redirect(url_for("login_page"))

    if not userinfo or "sub" not in userinfo:
        flash("Google login failed. Please try again.", "danger")
        return redirect(url_for("login_page"))

    google_sub = userinfo.get("sub")
    email = (userinfo.get("email") or "").lower()
    name = userinfo.get("name") or ""
    picture = userinfo.get("picture") or ""

    root = db.reference("/")
    oauth_ref = root.child("oauth_users").child(google_sub)
    oauth_ref.update({
        "email": email,
        "name": name,
        "picture": picture,
        "provider": "google"
    })

    session["username"] = email or name or google_sub
    session["auth_provider"] = "google"
    flash("Logged in with Google!", "success")
    return redirect(url_for("home1"))

# ---------------- Summarizer API (chunking + audio) ----------------
@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # BART input is limited; chunk long transcripts for stability
    words = text.split()
    chunk_size = 700  # adjust as needed
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    partials = []
    for ch in chunks:
        s = summarizer(ch, max_length=130, min_length=50, do_sample=False)[0]['summary_text']
        partials.append(s)

    combined = " ".join(partials)
    final_summary = summarizer(combined, max_length=160, min_length=60, do_sample=False)[0]['summary_text']

    # TTS (overwrites same file each time)
    os.makedirs(app.static_folder, exist_ok=True)
    audio_path_abs = os.path.join(app.static_folder, "summary.mp3")
    tts = gTTS(final_summary)
    tts.save(audio_path_abs)

    return jsonify({"summary": final_summary, "audio_file": "/static/summary.mp3"})

# ---------------- YouTube Video -> Text (download audio + whisper) ----------------
@app.route("/vedio", methods=["POST"])
def vedio_transcribe():
    """
    Accepts JSON: { "url": "https://www.youtube.com/watch?v=..." }
    Downloads audio via yt-dlp, transcribes with whisper, and returns JSON:
    { ok: True, video_id, transcript, formatted_html, plain_text }
    """
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "No URL provided"}), 400

    # Extract video id (robust)
    try:
        video_id = extract_video_id(url)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid YouTube URL or ID: {e}"}), 400

    # Prepare temp dir
    tmpdir = tempfile.mkdtemp(prefix="yttrans_")
    try:
        # Download audio
        try:
            audio_path, info = download_audio_from_youtube(url, tmpdir)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Audio download failed: {e}"}), 500

        # Check whisper model
        if whisper_model is None:
            return jsonify({"ok": False, "error": "Whisper model not loaded on server. Check WHISPER_MODEL and model installation."}), 500

        # Transcribe (this may take time)
        try:
            res = whisper_model.transcribe(audio_path)
            text = res.get('text', '').strip()
        except Exception as e:
            return jsonify({"ok": False, "error": f"Transcription failed: {e}"}), 500

        if not text:
            return jsonify({"ok": False, "error": "Transcription produced no text."}), 500

        formatted = format_transcript(text)

        return jsonify({
            "ok": True,
            "video_id": video_id,
            "transcript": text,
            "formatted_html": formatted.get("html"),
            "plain_text": formatted.get("plain"),
        })

    finally:
        # Cleanup tempdir
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

# ---------------- Newsletter Subscribe (optional) ----------------
@app.post("/subscribe")
def subscribe():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()

    if not EMAIL_RE.match(email):
        return jsonify({"ok": False, "error": "invalid_email"}), 400

    try:
        root = db.reference("/")
        idx_ref = root.child("newsletter_index").child(email_key(email))
        if idx_ref.get():
            return jsonify({"ok": True, "duplicate": True}), 200

        ts = int(time.time() * 1000)
        record = {
            "email": email,
            "ts": ts,
            "ua": request.headers.get("User-Agent", ""),
            "ip": request.headers.get("X-Forwarded-For", request.remote_addr),
            "source": "footer_form"
        }
        subs_ref = root.child("newsletter_subscriptions").push(record)
        idx_ref.set(True)
        return jsonify({"ok": True, "id": subs_ref.key}), 200
    except Exception:
        return jsonify({"ok": False, "error": "server_error"}), 500

# ---------------- Run (auto-open splash in browser) ----------------
def _open_browser():
    webbrowser.open("http://127.0.0.1:5000/")

if __name__ == "__main__":
    os.makedirs(app.static_folder, exist_ok=True)
    threading.Timer(1.0, _open_browser).start()
    app.run(debug=True, use_reloader=False)
