"""
app.py
------
Smart Phishing URL Detector — Streamlit frontend + backend in one file.

Detection flow:
  1. Rule-based: URL contains '@' OR more than 3 dots  -> "Suspicious"
  2. Otherwise:  Multinomial Naive Bayes model (see model.py)
                 -> "Safe" or "Phishing" with a confidence score.

Run:
    pip install -r requirements.txt
    streamlit run app.py
Then open the URL Streamlit prints (usually http://localhost:8501).
"""

from urllib.parse import urlparse
import streamlit as st

from model import predict_url


# ---------- Page setup ----------
st.set_page_config(
    page_title="Smart Phishing URL Detector",
    page_icon="🛡️",
    layout="centered",
)

# ---------- Custom dark theme + animations ----------
st.markdown("""
<style>
:root {
  --bg: #0a0e1a; --bg2: #0f1424;
  --text: #e7ecf5; --muted: #9aa6c2;
  --primary: #2dd4bf; --accent: #a78bfa;
  --safe: #22c55e; --suspicious: #facc15; --phishing: #ef4444;
  --border: rgba(120,140,200,0.18);
}
html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse at top, rgba(167,139,250,0.18), transparent 60%),
    radial-gradient(ellipse at bottom, rgba(45,212,191,0.15), transparent 55%),
    linear-gradient(180deg, var(--bg2), var(--bg)) !important;
  color: var(--text) !important;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3rem; max-width: 760px; }

.header { text-align: center; margin-bottom: 24px; animation: floatUp .6s ease both; }
.badge {
  display:inline-flex; align-items:center; justify-content:center;
  width:64px; height:64px; border-radius:18px; font-size:28px;
  background: rgba(45,212,191,0.12); border:1px solid rgba(45,212,191,0.35);
  margin-bottom:16px; animation: pulseRing 2s ease-out infinite;
}
.eyebrow { font-size:11px; letter-spacing:.3em; text-transform:uppercase; color:var(--muted); }
.title { font-size: clamp(32px, 5vw, 52px); font-weight:800; letter-spacing:-.02em; margin:8px 0 12px; }
.gradient {
  background: linear-gradient(135deg, var(--primary), #60a5fa);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.subtitle { color: var(--muted); max-width: 540px; margin: 0 auto; line-height:1.6; }

/* Input + button */
.stTextInput > div > div > input {
  background: rgba(15,20,36,0.7) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 12px !important;
  height: 52px; font-family: ui-monospace, 'JetBrains Mono', monospace;
}
.stTextInput > div > div > input:focus {
  border-color: rgba(45,212,191,0.6) !important;
  box-shadow: 0 0 0 4px rgba(45,212,191,0.15) !important;
}
.stButton > button {
  width: 100%; height: 48px; border: none; border-radius: 12px;
  font-weight: 700; letter-spacing:.02em; color:#061018;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  box-shadow: 0 10px 30px -10px rgba(45,212,191,0.6);
  transition: transform .15s ease, box-shadow .2s ease;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 18px 40px -12px rgba(45,212,191,0.7); }
.stButton > button:active { transform: translateY(0) scale(.99); }

/* Result card */
.result {
  margin-top: 18px; padding: 22px; border-radius: 14px;
  border: 1px solid var(--border); background: rgba(15,20,36,.55);
  animation: floatUp .4s ease both;
}
.result.safe       { box-shadow: 0 0 60px -10px rgba(34,197,94,.55);  border-color: rgba(34,197,94,.45); }
.result.suspicious { box-shadow: 0 0 60px -10px rgba(250,204,21,.55); border-color: rgba(250,204,21,.45); }
.result.phishing   { box-shadow: 0 0 60px -10px rgba(239,68,68,.6);   border-color: rgba(239,68,68,.5); }
.chip {
  display:inline-block; padding:4px 10px; border-radius:999px;
  font-size:11px; font-weight:800; letter-spacing:.12em; text-transform:uppercase;
}
.chip.safe       { background: var(--safe);       color:#06210f; }
.chip.suspicious { background: var(--suspicious); color:#2a1f00; }
.chip.phishing   { background: var(--phishing);   color:#fff; }
.meta { font-size:12px; color:var(--muted); margin-left:8px; }
.result h2 { margin: 10px 0 6px; font-size:24px; font-weight:800; }
.result.safe       h2 { color: var(--safe); }
.result.suspicious h2 { color: var(--suspicious); }
.result.phishing   h2 { color: var(--phishing); }
.result p.reason { color: var(--muted); margin: 0 0 12px; line-height: 1.55; }
.result .url {
  background: rgba(20,26,45,.7); border:1px solid var(--border);
  border-radius:10px; padding:10px 12px;
  font-family: ui-monospace, 'JetBrains Mono', monospace; font-size:13px; word-break:break-all;
}

/* Legend */
.legend { display:grid; gap:10px; grid-template-columns: repeat(3, 1fr); margin-top: 22px; }
.legend .item {
  display:flex; align-items:center; gap:8px; font-size:12px; color:var(--muted);
  background: rgba(20,26,45,.5); border:1px solid var(--border);
  padding:10px 12px; border-radius:12px;
}
.dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.dot.safe       { background: var(--safe);       box-shadow: 0 0 12px var(--safe); }
.dot.suspicious { background: var(--suspicious); box-shadow: 0 0 12px var(--suspicious); }
.dot.phishing   { background: var(--phishing);   box-shadow: 0 0 12px var(--phishing); }
@media (max-width: 560px) { .legend { grid-template-columns: 1fr; } }

@keyframes floatUp { from { opacity:0; transform: translateY(14px); } to { opacity:1; transform: translateY(0); } }
@keyframes pulseRing {
  0%   { box-shadow: 0 0 0 0 rgba(45,212,191,.5); }
  100% { box-shadow: 0 0 0 22px rgba(45,212,191,0); }
}
</style>
""", unsafe_allow_html=True)


# ---------- Helpers ----------
def is_valid_url(url: str) -> bool:
    if not url or len(url) > 2048:
        return False
    candidate = url if "://" in url else "http://" + url
    try:
        p = urlparse(candidate)
        return bool(p.netloc) and "." in p.netloc
    except Exception:
        return False


def analyze(url: str):
    """Rule-based first, then ML."""
    dots = url.count(".")
    if "@" in url or dots > 3:
        why = []
        if "@" in url: why.append("contains '@' symbol")
        if dots > 3:  why.append(f"has {dots} dots (more than 3)")
        return {
            "verdict": "Suspicious",
            "reason":  "URL " + " and ".join(why) + ".",
            "confidence": None,
            "source": "Rule-based heuristics",
        }
    label, conf = predict_url(url)
    return {
        "verdict": label,
        "reason": "Naive Bayes model recognized phishing-like patterns."
                  if label == "Phishing"
                  else "Naive Bayes model found no phishing patterns.",
        "confidence": conf,
        "source": "Naive Bayes ML model",
    }


# ---------- Header ----------
st.markdown("""
<div class="header">
  <div class="badge">🛡️</div>
  <div class="eyebrow">Cybersecurity · Threat Intelligence</div>
  <div class="title">Smart Phishing <span class="gradient">URL Detector</span></div>
  <p class="subtitle">
    Paste any link to scan it instantly. We combine
    <b style="color:var(--text)">heuristic rules</b> with a
    <b style="color:var(--text)">Naive Bayes</b> ML model
    to flag phishing attempts in real time.
  </p>
</div>
""", unsafe_allow_html=True)


# ---------- Form ----------
with st.form("scan", clear_on_submit=False):
    url = st.text_input(
        "Enter a URL to scan",
        placeholder="https://example.com/login",
        max_chars=2048,
    )
    submitted = st.form_submit_button("🔎  Check URL")

# ---------- Result ----------
if submitted:
    url = (url or "").strip()
    if not url:
        st.error("Please enter a URL to scan.")
    elif not is_valid_url(url):
        st.error("That doesn't look like a valid URL.")
    else:
        r = analyze(url)
        v = r["verdict"].lower()
        conf_html = (
            f'<span class="meta">{r["confidence"]*100:.1f}% confidence</span>'
            if r["confidence"] is not None else ""
        )
        title = {
            "safe": "URL appears safe",
            "suspicious": "Use caution",
            "phishing": "Likely phishing",
        }[v]
        st.markdown(f"""
        <div class="result {v}">
          <div>
            <span class="chip {v}">{r["verdict"]}</span>
            <span class="meta">via {r["source"]}</span>
            {conf_html}
          </div>
          <h2>{title}</h2>
          <p class="reason">{r["reason"]}</p>
          <div class="url">{url}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------- Legend ----------
st.markdown("""
<div class="legend">
  <div class="item"><span class="dot safe"></span> Safe — No phishing patterns</div>
  <div class="item"><span class="dot suspicious"></span> Suspicious — Heuristic red flags</div>
  <div class="item"><span class="dot phishing"></span> Phishing — Matches phishing patterns</div>
</div>
<p style="text-align:center; color:var(--muted); font-size:12px; margin-top:28px;">
  Educational demo · Not a substitute for professional security tools
</p>
""", unsafe_allow_html=True)
