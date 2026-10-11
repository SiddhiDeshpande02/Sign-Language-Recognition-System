import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignSense · ASL Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0b0e11;
    --surface:   #13181f;
    --border:    #1e2630;
    --accent:    #00e5a0;
    --accent2:   #0af;
    --muted:     #4a5568;
    --text:      #e8edf3;
    --text-soft: #8898aa;
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.02em;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="metric-container"] label {
    color: var(--text-soft) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 2rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Info boxes ── */
.stInfo, [data-testid="stInfo"] {
    background: rgba(0,229,160,0.06) !important;
    border: 1px solid rgba(0,229,160,0.25) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Letter display ── */
.letter-display {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin-top: 12px;
}
.letter-big {
    font-family: 'Space Mono', monospace;
    font-size: 9rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    text-shadow: 0 0 60px rgba(0,229,160,0.35);
    margin: 0;
}
.letter-label {
    color: var(--text-soft);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 8px;
}

/* ── Alphabet grid ── */
.alpha-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(36px, 1fr));
    gap: 6px;
    margin-top: 12px;
}
.alpha-cell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    text-align: center;
    padding: 6px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-soft);
    transition: all 0.2s;
}
.alpha-cell.active {
    background: rgba(0,229,160,0.15);
    border-color: var(--accent);
    color: var(--accent);
    box-shadow: 0 0 12px rgba(0,229,160,0.2);
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,229,160,0.1);
    border: 1px solid rgba(0,229,160,0.3);
    border-radius: 100px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: var(--accent);
    font-family: 'Space Mono', monospace;
}
.dot {
    width: 7px; height: 7px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
}

/* ── Sidebar brand ── */
.brand {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.03em;
    margin-bottom: 4px;
}
.brand span { color: var(--accent); }
.tagline {
    color: var(--text-soft);
    font-size: 0.78rem;
    margin-bottom: 24px;
}

/* ── WebRTC video ── */
.stVideo video,
video {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* ── TTS toggle ── */
.tts-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 10px;
}
.tts-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-soft);
}
.tts-label span { color: var(--accent); font-size: 1rem; }

/* ── Speaking flash ── */
@keyframes flash {
    0%   { box-shadow: 0 0 0 rgba(0,229,160,0); }
    40%  { box-shadow: 0 0 28px rgba(0,229,160,0.55); }
    100% { box-shadow: 0 0 0 rgba(0,229,160,0); }
}
.letter-display.speaking { animation: flash 0.6s ease-out; }

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
    background: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)


# ── MediaPipe + model setup ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists('./model.p'):
        return None
    with open('./model.p', 'rb') as f:
        return pickle.load(f)['model']

@st.cache_resource
def get_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

LABELS = {i: chr(65 + i) for i in range(26)}   # 0→A … 25→Z
mp_hands_module = mp.solutions.hands
mp_drawing       = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

model = load_model()

# ── Shared state ──────────────────────────────────────────────────────────────
if "last_letter"     not in st.session_state: st.session_state.last_letter     = "–"
if "confidence"      not in st.session_state: st.session_state.confidence      = 0.0
if "hand_detected"   not in st.session_state: st.session_state.hand_detected   = False
if "letter_history"  not in st.session_state: st.session_state.letter_history  = []
if "tts_enabled"     not in st.session_state: st.session_state.tts_enabled     = True
if "last_spoken"     not in st.session_state: st.session_state.last_spoken     = ""
if "speaking"        not in st.session_state: st.session_state.speaking        = False


# ── Video processor ───────────────────────────────────────────────────────────
class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands_module.Hands(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.predicted = "–"
        self.conf      = 0.0
        self.detected  = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(img_rgb)
        self.detected = False

        if results.multi_hand_landmarks and model is not None:
            self.detected = True
            x_, y_, data_aux = [], [], []

            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hand_lm,
                    mp_hands_module.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                for lm in hand_lm.landmark:
                    x_.append(lm.x); y_.append(lm.y)
                for lm in hand_lm.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            if len(data_aux) == 42:
                pred       = model.predict([np.asarray(data_aux)])
                proba      = model.predict_proba([np.asarray(data_aux)])
                self.conf  = float(np.max(proba))
                letter     = LABELS[int(pred[0])]
                self.predicted = letter

                H, W, _ = img.shape
                x1 = max(0, int(min(x_) * W) - 20)
                y1 = max(0, int(min(y_) * H) - 20)
                x2 = min(W, int(max(x_) * W) + 20)
                y2 = min(H, int(max(y_) * H) + 20)

                # Bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 229, 160), 2)

                # Badge background
                badge_w, badge_h = 60, 42
                bx1, by1 = x1, max(0, y1 - badge_h)
                bx2, by2 = x1 + badge_w, y1
                overlay = img.copy()
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 229, 160), -1)
                cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

                # Letter text
                cv2.putText(img, letter, (bx1 + 6, by2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)

                # Confidence bar
                bar_w = int((x2 - x1) * self.conf)
                cv2.rectangle(img, (x1, y2 + 4), (x2, y2 + 10), (30, 40, 50), -1)
                cv2.rectangle(img, (x1, y2 + 4), (x1 + bar_w, y2 + 10), (0, 229, 160), -1)

        # Watermark
        cv2.putText(img, "SignSense", (10, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 80, 100), 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">Sign<span>Sense</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">ASL Alphabet · Real-time Detection</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Model status
    if model is not None:
        st.markdown('<div class="status-pill"><div class="dot"></div> Model loaded</div>',
                    unsafe_allow_html=True)
    else:
        st.error("⚠️ model.p not found — run `train_classifier.py` first.")

    st.markdown("---")
    st.markdown("#### 🔢 Detection Stats")

    col_a, col_b = st.columns(2)
    with col_a:
        letter_ph = st.empty()
    with col_b:
        conf_ph = st.empty()

    hand_ph = st.empty()

    st.markdown("---")
    st.markdown("#### 🔤 Alphabet")

    # Live alphabet grid placeholder
    alpha_ph = st.empty()

    st.markdown("---")
    st.markdown("#### 📋 Session log")
    log_ph = st.empty()

    if st.button("Clear log"):
        st.session_state.letter_history = []

    st.markdown("---")
    st.markdown("#### 🔊 Text-to-Speech")

    tts_toggle = st.toggle("Speak detected letter", value=st.session_state.tts_enabled)
    st.session_state.tts_enabled = tts_toggle

    if tts_toggle:
        tts_rate  = st.slider("Speed", 0.5, 2.0, 0.85, 0.05,
                              help="Speech rate: 1.0 = normal, <1 slower, >1 faster")
        tts_pitch = st.slider("Pitch", 0.5, 2.0, 1.0, 0.1,
                              help="Voice pitch: 1.0 = default")
    else:
        tts_rate, tts_pitch = 0.85, 1.0

    # Placeholder for the invisible TTS component
    tts_ph = st.empty()

    st.markdown("---")
    st.caption("Built with MediaPipe · scikit-learn · Streamlit")


# ── TTS helper (browser Web Speech API) ──────────────────────────────────────
def speak(placeholder, letter: str, rate: float = 0.85, pitch: float = 1.0):
    """Inject a JS snippet into a placeholder that speaks the letter via Web Speech API."""
    uid = int(time.time() * 1000)   # unique uid forces re-render every call
    placeholder.empty()
    with placeholder:
        components.html(f"""
        <script>
        (function() {{
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance("{letter}");
            u.rate   = {rate};
            u.pitch  = {pitch};
            u.volume = 1;
            // Prefer a local English voice for clarity
            const trySpeak = () => {{
                const voices = window.speechSynthesis.getVoices();
                const en = voices.find(v => v.lang.startsWith('en') && v.localService);
                if (en) u.voice = en;
                window.speechSynthesis.speak(u);
            }};
            // Voices may not be loaded yet on first run
            if (window.speechSynthesis.getVoices().length === 0) {{
                window.speechSynthesis.onvoiceschanged = trySpeak;
            }} else {{
                trySpeak();
            }}
        }})();
        // ts={uid}
        </script>
        """, height=0, scrolling=False)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## Sign Language Detector")
st.markdown("Show your hand to the camera — the model will recognise **A–Z** in real time.")

st.markdown("---")

col_video, col_info = st.columns([2, 1], gap="large")

with col_video:
    if model is None:
        st.warning("Load a trained model (`model.p`) to enable live detection.")
    
    ctx = webrtc_streamer(
        key="sign-lang",
        video_processor_factory=SignProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    st.markdown("#### Detected Sign")
    letter_big_ph = st.empty()
    st.markdown("---")
    st.markdown("#### How to use")
    st.markdown("""
1. Allow camera access  
2. Hold one hand clearly in frame  
3. Keep background plain if possible  
4. Form any ASL letter — result appears instantly
    """)
    st.markdown("---")
    st.info("💡 For best accuracy, ensure good lighting and keep your hand centred.")


# ── Live update loop ──────────────────────────────────────────────────────────
def render_alpha_grid(active: str) -> str:
    cells = ""
    for i in range(26):
        ch   = chr(65 + i)
        cls  = "alpha-cell active" if ch == active else "alpha-cell"
        cells += f'<div class="{cls}">{ch}</div>'
    return f'<div class="alpha-grid">{cells}</div>'

if ctx.state.playing and ctx.video_processor:
    proc = ctx.video_processor

    while ctx.state.playing:
        letter = proc.predicted
        conf   = proc.conf
        det    = proc.detected

        # ── TTS: speak when letter changes ──────────────────────────────────
        if (det and letter != "–" and
                st.session_state.tts_enabled and
                letter != st.session_state.last_spoken):
            speak(tts_ph, letter, rate=tts_rate, pitch=tts_pitch)
            st.session_state.last_spoken = letter

        # Reset last_spoken when hand leaves frame so next detection speaks again
        if not det:
            st.session_state.last_spoken = ""

        # Update session state
        if det and letter != "–":
            if (not st.session_state.letter_history or
                    st.session_state.letter_history[-1] != letter):
                st.session_state.letter_history.append(letter)
                st.session_state.letter_history = st.session_state.letter_history[-40:]

        # Sidebar metrics
        letter_ph.metric("Letter", letter if det else "–")
        conf_ph.metric("Confidence", f"{conf*100:.0f}%" if det else "–")
        hand_ph.info("✋ Hand detected" if det else "🖐️ Waiting for hand…")

        # Alphabet grid
        alpha_ph.markdown(render_alpha_grid(letter if det else ""), unsafe_allow_html=True)

        # Session log
        log_text = " ".join(st.session_state.letter_history[-20:]) or "—"
        log_ph.code(log_text)

        # Big letter in main area
        is_speaking = (det and st.session_state.tts_enabled and
                       letter == st.session_state.last_spoken)
        speaking_class = "letter-display speaking" if is_speaking else "letter-display"
        tts_badge = (
            '<span style="font-size:1rem;margin-left:8px;opacity:0.7;">🔊</span>'
            if is_speaking else ""
        )
        letter_big_ph.markdown(f"""
<div class="{speaking_class}">
  <p class="letter-big">{letter if det else '–'}</p>
  <p class="letter-label">
    {'Recognised · ' + f'{conf*100:.0f}% confidence' if det else 'No hand in frame'}
    {tts_badge}
  </p>
</div>
""", unsafe_allow_html=True)

        time.sleep(0.08)   # ~12 fps refresh
else:
    # Static placeholder when camera is off
    letter_big_ph.markdown("""
<div class="letter-display">
  <p class="letter-big" style="color:#1e2630;">–</p>
  <p class="letter-label">Start the camera to begin</p>
</div>
""", unsafe_allow_html=True)
    alpha_ph.markdown(render_alpha_grid(""), unsafe_allow_html=True)
    log_ph.code("—")
    letter_ph.metric("Letter", "–")
    conf_ph.metric("Confidence", "–")
    hand_ph.info("🖐️ Camera is off")