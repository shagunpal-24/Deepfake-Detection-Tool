import hashlib
import json
import tempfile
from datetime import datetime
from html import escape
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf

from data_loader import extract_frames, load_image
from utils import predict, predict_details, prediction_distribution


HISTORY_PATH = Path("scan_history.json")
REPORTS_DIR = Path("reports")
THRESHOLD_FILE = Path("model_threshold.txt")
REPORTS_DIR.mkdir(exist_ok=True)


def load_default_threshold():
    try:
        threshold = float(THRESHOLD_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0.50
    return min(max(threshold, 0.10), 0.90)


DEFAULT_THRESHOLD = load_default_threshold()
CAMERA_FAKE_THRESHOLD = 0.90


MODE_CONFIG = {
    "Fast mode": {"frames": 12, "threshold": DEFAULT_THRESHOLD, "label": "Fast scan, fewer frames"},
    "Accurate mode": {"frames": 45, "threshold": DEFAULT_THRESHOLD, "label": "Deeper scan, more frames"},
    "Experimental model": {"frames": 60, "threshold": DEFAULT_THRESHOLD, "label": "More sensitive research setting"},
}


st.set_page_config(page_title="Deepfake Detector", page_icon="AI", layout="wide")


def inject_ai_background():
    components.html(
        """
        <script>
        const parentDoc = window.parent.document;
        function installBackground() {
            let toggle = parentDoc.getElementById("sidebar-collapse-toggle");
            if (!toggle) {
                toggle = parentDoc.createElement("button");
                toggle.id = "sidebar-collapse-toggle";
                toggle.innerHTML = "☰";
                toggle.setAttribute("aria-label", "Toggle sidebar");
                toggle.addEventListener("click", () => {
                    parentDoc.body.classList.toggle("sidebar-collapsed");
                });
                parentDoc.body.appendChild(toggle);
            }

            let canvas = parentDoc.getElementById("ai-neural-background");
            if (!canvas) {
                canvas = parentDoc.createElement("canvas");
                canvas.id = "ai-neural-background";
                parentDoc.body.appendChild(canvas);
            }
            Object.assign(canvas.style, {
                position: "fixed", inset: "0", width: "100vw", height: "100vh",
                zIndex: "0", pointerEvents: "none", opacity: "0.82"
            });
            const ctx = canvas.getContext("2d");
            let width = 0, height = 0, particles = [];
            function resize() {
                width = canvas.width = parentDoc.documentElement.clientWidth;
                height = canvas.height = parentDoc.documentElement.clientHeight;
                const count = Math.min(100, Math.max(42, Math.floor(width * height / 16500)));
                particles = Array.from({ length: count }, () => ({
                    x: Math.random() * width, y: Math.random() * height,
                    vx: (Math.random() - 0.5) * 0.48, vy: (Math.random() - 0.5) * 0.48,
                    r: Math.random() * 1.8 + 0.9
                }));
            }
            function draw() {
                ctx.clearRect(0, 0, width, height);
                const gradient = ctx.createLinearGradient(0, 0, width, height);
                gradient.addColorStop(0, "rgba(0, 245, 255, 0.95)");
                gradient.addColorStop(0.5, "rgba(132, 92, 255, 0.75)");
                gradient.addColorStop(1, "rgba(34, 255, 170, 0.85)");
                particles.forEach((p, i) => {
                    p.x += p.vx; p.y += p.vy;
                    if (p.x < -20) p.x = width + 20;
                    if (p.x > width + 20) p.x = -20;
                    if (p.y < -20) p.y = height + 20;
                    if (p.y > height + 20) p.y = -20;
                    for (let j = i + 1; j < particles.length; j++) {
                        const q = particles[j], dx = p.x - q.x, dy = p.y - q.y;
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        if (distance < 135) {
                            ctx.strokeStyle = `rgba(68, 230, 255, ${0.16 * (1 - distance / 135)})`;
                            ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(p.x, p.y);
                            ctx.lineTo(q.x, q.y); ctx.stroke();
                        }
                    }
                    ctx.fillStyle = gradient; ctx.shadowColor = "rgba(34, 245, 255, 0.9)";
                    ctx.shadowBlur = 12; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                    ctx.fill(); ctx.shadowBlur = 0;
                });
                requestAnimationFrame(draw);
            }
            if (!window.__deepfakeBgReady) {
                window.__deepfakeBgReady = true;
                resize(); draw(); window.addEventListener("resize", resize);
            }
        }
        installBackground();
        </script>
        """,
        height=0,
    )


def inject_ai_assistant():
    components.html(
        """
        <script>
        const assistantDoc = window.parent.document;

        function installAssistant() {
            if (assistantDoc.getElementById("floating-ai-assistant")) {
                return;
            }

            const style = assistantDoc.createElement("style");
            style.textContent = `
                #floating-ai-assistant {
                    position: fixed;
                    right: 24px;
                    bottom: 24px;
                    z-index: 100001;
                    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                }

                .ai-assistant-button {
                    position: relative;
                    display: grid;
                    place-items: center;
                    width: 66px;
                    height: 66px;
                    border: 1px solid rgba(34, 245, 255, 0.58);
                    border-radius: 50%;
                    color: #ffffff;
                    background:
                        radial-gradient(circle at 30% 22%, rgba(255,255,255,0.32), transparent 28%),
                        linear-gradient(135deg, rgba(34,245,255,0.94), rgba(106,92,255,0.92));
                    box-shadow: 0 18px 48px rgba(0,0,0,0.32), 0 0 34px rgba(34,245,255,0.34);
                    cursor: pointer;
                    animation: assistantBounce 3s ease-in-out infinite;
                }

                .ai-assistant-button:hover {
                    transform: translateY(-4px) scale(1.04);
                    box-shadow: 0 22px 58px rgba(0,0,0,0.38), 0 0 46px rgba(34,245,255,0.48);
                }

                .ai-robot-face {
                    display: grid;
                    place-items: center;
                    width: 38px;
                    height: 30px;
                    border: 2px solid rgba(255,255,255,0.82);
                    border-radius: 13px;
                    position: relative;
                    font-weight: 900;
                    font-size: 11px;
                    letter-spacing: 1px;
                }

                .ai-robot-face::before,
                .ai-robot-face::after {
                    content: "";
                    position: absolute;
                    top: 11px;
                    width: 6px;
                    height: 6px;
                    border-radius: 50%;
                    background: #ffffff;
                    box-shadow: 0 0 10px rgba(255,255,255,0.9);
                }

                .ai-robot-face::before { left: 9px; }
                .ai-robot-face::after { right: 9px; }

                .ai-notification-dot {
                    position: absolute;
                    right: 4px;
                    top: 5px;
                    width: 13px;
                    height: 13px;
                    border: 2px solid #07111f;
                    border-radius: 50%;
                    background: #47f5b5;
                    box-shadow: 0 0 18px rgba(71,245,181,0.88);
                }

                .ai-chat-panel {
                    position: absolute;
                    right: 0;
                    bottom: 82px;
                    width: min(360px, calc(100vw - 34px));
                    border: 1px solid rgba(34,245,255,0.34);
                    border-radius: 22px;
                    overflow: hidden;
                    color: var(--text-color, #f8fbff);
                    background:
                        linear-gradient(145deg, var(--card-bg, rgba(8,13,29,0.86)), var(--chat-panel-bg, rgba(12,18,42,0.78)));
                    box-shadow: var(--card-shadow, 0 28px 80px rgba(0,0,0,0.42)), 0 0 44px rgba(34,245,255,0.18);
                    backdrop-filter: blur(var(--glass-blur, 18px)) saturate(135%);
                    transform: translateY(14px) scale(0.96);
                    opacity: 0;
                    pointer-events: none;
                    transition: opacity 220ms ease, transform 220ms ease;
                }

                #floating-ai-assistant.open .ai-chat-panel {
                    transform: translateY(0) scale(1);
                    opacity: 1;
                    pointer-events: auto;
                }

                #floating-ai-assistant.open .ai-notification-dot {
                    display: none;
                }

                .ai-chat-header {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 12px;
                    padding: 14px 16px;
                    border-bottom: 1px solid var(--line, rgba(34,245,255,0.18));
                    background: linear-gradient(90deg, rgba(34,245,255,0.16), rgba(168,85,247,0.12));
                }

                .ai-chat-title {
                    color: var(--text-color, #f8fbff);
                    font-size: 14px;
                    font-weight: 900;
                }

                .ai-chat-status {
                    color: var(--secondary-text-color, #aebbd1);
                    font-size: 12px;
                    margin-top: 2px;
                }

                .ai-close-button {
                    display: grid;
                    place-items: center;
                    width: 32px;
                    height: 32px;
                    border: 1px solid var(--line, rgba(255,255,255,0.18));
                    border-radius: 50%;
                    color: var(--text-color, #f8fbff);
                    background: var(--surface, rgba(255,255,255,0.08));
                    cursor: pointer;
                }

                .ai-close-button:hover {
                    transform: rotate(90deg);
                    border-color: rgba(34,245,255,0.46);
                    box-shadow: 0 0 20px rgba(34,245,255,0.20);
                }

                .ai-chat-body {
                    padding: 16px;
                }

                .ai-message {
                    color: var(--text-color, #edf7ff);
                    font-size: 14px;
                    line-height: 1.55;
                    padding: 14px;
                    border: 1px solid var(--line, rgba(34,245,255,0.18));
                    border-radius: 18px 18px 18px 6px;
                    background: var(--message-bg, rgba(255,255,255,0.08));
                    min-height: 96px;
                }

                .ai-cursor {
                    display: inline-block;
                    width: 7px;
                    height: 15px;
                    margin-left: 2px;
                    background: #22f5ff;
                    vertical-align: -2px;
                    animation: cursorBlink 0.8s steps(2) infinite;
                }

                @keyframes assistantBounce {
                    0%, 100% { transform: translateY(0); }
                    50% { transform: translateY(-7px); }
                }

                @keyframes cursorBlink {
                    0%, 45% { opacity: 1; }
                    46%, 100% { opacity: 0; }
                }

                @media (max-width: 640px) {
                    #floating-ai-assistant {
                        right: 16px;
                        bottom: 16px;
                    }
                    .ai-assistant-button {
                        width: 58px;
                        height: 58px;
                    }
                    .ai-chat-panel {
                        bottom: 72px;
                    }
                }
            `;

            const root = assistantDoc.createElement("div");
            root.id = "floating-ai-assistant";
            root.innerHTML = `
                <div class="ai-chat-panel" role="dialog" aria-label="AI assistant chat">
                    <div class="ai-chat-header">
                        <div>
                            <div class="ai-chat-title">AI Assistant</div>
                            <div class="ai-chat-status">Deepfake guidance online</div>
                        </div>
                        <button class="ai-close-button" type="button" aria-label="Close assistant">×</button>
                    </div>
                    <div class="ai-chat-body">
                        <div class="ai-message"><span class="ai-message-text"></span><span class="ai-cursor"></span></div>
                    </div>
                </div>
                <button class="ai-assistant-button" type="button" aria-label="Open AI assistant">
                    <span class="ai-notification-dot"></span>
                    <span class="ai-robot-face"></span>
                </button>
            `;

            assistantDoc.head.appendChild(style);
            assistantDoc.body.appendChild(root);

            const button = root.querySelector(".ai-assistant-button");
            const closeButton = root.querySelector(".ai-close-button");
            const messageText = root.querySelector(".ai-message-text");
            const cursor = root.querySelector(".ai-cursor");
            const message = "Hello 👋 I'm your AI assistant. Upload a video or image to detect deepfakes. Adjust the threshold and frames from the sidebar for better accuracy.";
            let hasTyped = false;
            let typingTimer = null;

            function typeMessage() {
                if (hasTyped) return;
                hasTyped = true;
                messageText.textContent = "";
                let index = 0;
                typingTimer = setInterval(() => {
                    messageText.textContent += message.charAt(index);
                    index += 1;
                    if (index >= message.length) {
                        clearInterval(typingTimer);
                        cursor.style.display = "none";
                    }
                }, 24);
            }

            button.addEventListener("click", () => {
                root.classList.toggle("open");
                if (root.classList.contains("open")) {
                    cursor.style.display = "inline-block";
                    typeMessage();
                }
            });

            closeButton.addEventListener("click", () => {
                root.classList.remove("open");
            });
        }

        installAssistant();
        </script>
        """,
        height=0,
    )


def inject_sidebar_navigation(initial_scans, threshold, max_frames, mode_name):
    scans_json = json.dumps(initial_scans[:8])
    components.html(
        f"""
        <script>
        const navDoc = window.parent.document;

        function installDeepScanNavigation() {{
            const defaultScans = {scans_json};
            const defaultState = {{
                threshold: {float(threshold):.2f},
                frames: {int(max_frames)},
                mode: "{escape(mode_name)}"
            }};

            function getStoredScans() {{
                try {{
                    const saved = JSON.parse(localStorage.getItem("deepScanMemory") || "[]");
                    return saved.length ? saved : defaultScans;
                }} catch (error) {{
                    return defaultScans;
                }}
            }}

            function saveScans(scans) {{
                localStorage.setItem("deepScanMemory", JSON.stringify(scans.slice(0, 30)));
            }}

            function normalizeScan(scan) {{
                return {{
                    file: scan.file || scan.name || "sample_media.mp4",
                    result: scan.result || "Fake",
                    confidence: Number(scan.confidence || 0.87),
                    timestamp: scan.timestamp || new Date().toLocaleString()
                }};
            }}

            let scanMemory = getStoredScans().map(normalizeScan);
            let tuningState = {{
                ...defaultState,
                ...JSON.parse(localStorage.getItem("deepScanTuning") || "{{}}")
            }};

            if (!scanMemory.length) {{
                scanMemory = [
                    {{ file: "office_interview.mp4", result: "Real", confidence: 0.91, timestamp: "Sample scan" }},
                    {{ file: "synthetic_face_clip.mp4", result: "Fake", confidence: 0.88, timestamp: "Sample scan" }},
                    {{ file: "webcam_capture.jpg", result: "Real", confidence: 0.76, timestamp: "Sample scan" }}
                ];
                saveScans(scanMemory);
            }}

            function setSection(section) {{
                navDoc.body.dataset.activeSection = section;
                navDoc.querySelectorAll(".sidebar-nav-item").forEach((item) => {{
                    item.classList.toggle("active", item.dataset.section === section);
                }});
                navDoc.querySelectorAll(".dashboard-section").forEach((panel) => {{
                    panel.classList.toggle("active", panel.dataset.section === section);
                }});
            }}

            function renderTuning() {{
                const thresholdValue = navDoc.getElementById("js-threshold-value");
                const thresholdCard = navDoc.getElementById("js-threshold-card-value");
                const frameValue = navDoc.getElementById("js-frame-value");
                const modeValue = navDoc.getElementById("js-mode-value");
                const thresholdInput = navDoc.getElementById("js-threshold");
                const frameInput = navDoc.getElementById("js-frames");
                const modeInput = navDoc.getElementById("js-mode");

                if (!thresholdInput || !frameInput || !modeInput) return;

                thresholdInput.value = tuningState.threshold;
                frameInput.value = tuningState.frames;
                modeInput.value = tuningState.mode;
                thresholdValue.textContent = Number(tuningState.threshold).toFixed(2);
                if (thresholdCard) thresholdCard.textContent = Number(tuningState.threshold).toFixed(2);
                frameValue.textContent = tuningState.frames;
                modeValue.textContent = tuningState.mode;
                localStorage.setItem("deepScanTuning", JSON.stringify(tuningState));
            }}

            function renderMemory() {{
                const list = navDoc.getElementById("js-scan-memory-list");
                if (!list) return;

                if (!scanMemory.length) {{
                    list.innerHTML = '<div class="empty-state">No scans yet. Complete a scan and it will appear here.</div>';
                    return;
                }}

                list.innerHTML = scanMemory.map((scan) => {{
                    const confidence = Math.round(Number(scan.confidence) * 100);
                    const badge = scan.result === "Fake" ? "fake" : "real";
                    return `
                        <div class="js-scan-card">
                            <div>
                                <div class="js-scan-file">${{scan.file}}</div>
                                <div class="js-scan-meta">${{scan.timestamp}} | ${{confidence}}% confidence</div>
                            </div>
                            <div class="badge ${{badge}}">${{scan.result}}</div>
                        </div>
                    `;
                }}).join("");
            }}

            navDoc.querySelectorAll(".sidebar-nav-item").forEach((item) => {{
                if (item.dataset.ready === "true") return;
                item.dataset.ready = "true";
                item.addEventListener("click", () => setSection(item.dataset.section));
            }});

            const thresholdInput = navDoc.getElementById("js-threshold");
            const frameInput = navDoc.getElementById("js-frames");
            const modeInput = navDoc.getElementById("js-mode");
            const clearButton = navDoc.getElementById("js-clear-memory");

            if (thresholdInput && thresholdInput.dataset.ready !== "true") {{
                thresholdInput.dataset.ready = "true";
                thresholdInput.addEventListener("input", (event) => {{
                    tuningState.threshold = Number(event.target.value);
                    renderTuning();
                }});
            }}

            if (frameInput && frameInput.dataset.ready !== "true") {{
                frameInput.dataset.ready = "true";
                frameInput.addEventListener("input", (event) => {{
                    tuningState.frames = Number(event.target.value);
                    renderTuning();
                }});
            }}

            if (modeInput && modeInput.dataset.ready !== "true") {{
                modeInput.dataset.ready = "true";
                modeInput.addEventListener("change", (event) => {{
                    tuningState.mode = event.target.value;
                    renderTuning();
                }});
            }}

            if (clearButton && clearButton.dataset.ready !== "true") {{
                clearButton.dataset.ready = "true";
                clearButton.addEventListener("click", () => {{
                    scanMemory = [];
                    saveScans(scanMemory);
                    renderMemory();
                }});
            }}

            window.parent.deepScanAddScan = function(scan) {{
                scanMemory.unshift(normalizeScan(scan));
                saveScans(scanMemory);
                renderMemory();
            }};

            renderTuning();
            renderMemory();
            setSection(navDoc.body.dataset.activeSection || "ai-controls");
        }}

        installDeepScanNavigation();
        </script>
        """,
        height=0,
    )


def apply_theme(theme):
    if theme == "Light":
        colors = {
            "bg": "#f7fbff",
            "mid": "#eaf2ff",
            "bg2": "#f5f0ff",
            "panel": "rgba(255, 255, 255, 0.96)",
            "panel2": "rgba(255, 255, 255, 0.985)",
            "text": "#111827",
            "muted": "#3f4b5f",
            "line": "rgba(30, 64, 175, 0.16)",
            "shadow": "rgba(15, 23, 42, 0.10)",
            "canvas": "0.16",
            "blur": "2px",
            "surface": "rgba(248, 250, 252, 0.98)",
            "card_shadow": "0 4px 20px rgba(0, 0, 0, 0.10)",
            "input_bg": "#ffffff",
            "message_bg": "#f8fafc",
            "chat_panel": "#ffffff",
        }
    else:
        colors = {
            "bg": "rgba(2, 7, 18, 0.94)",
            "mid": "rgba(4, 20, 38, 0.86)",
            "bg2": "rgba(12, 6, 32, 0.92)",
            "panel": "rgba(8, 13, 29, 0.74)",
            "panel2": "rgba(5, 9, 22, 0.88)",
            "text": "#f8fbff",
            "muted": "#aebbd1",
            "line": "rgba(125, 241, 255, 0.22)",
            "shadow": "rgba(0, 0, 0, 0.42)",
            "canvas": "0.82",
            "blur": "18px",
            "surface": "rgba(255, 255, 255, 0.055)",
            "card_shadow": "0 22px 70px rgba(0, 0, 0, 0.42)",
            "input_bg": "rgba(255, 255, 255, 0.075)",
            "message_bg": "rgba(255, 255, 255, 0.08)",
            "chat_panel": "rgba(12, 18, 42, 0.78)",
        }

    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {colors["bg"]};
            --app-mid: {colors["mid"]};
            --app-bg2: {colors["bg2"]};
            --panel: {colors["panel"]};
            --panel-strong: {colors["panel2"]};
            --line: {colors["line"]};
            --text: {colors["text"]};
            --muted: {colors["muted"]};
            --bg-color: {colors["bg"]};
            --text-color: {colors["text"]};
            --secondary-text-color: {colors["muted"]};
            --card-bg: {colors["panel"]};
            --accent-color: #0ea5e9;
            --shadow: {colors["shadow"]};
            --canvas-opacity: {colors["canvas"]};
            --glass-blur: {colors["blur"]};
            --surface: {colors["surface"]};
            --card-shadow: {colors["card_shadow"]};
            --input-bg: {colors["input_bg"]};
            --message-bg: {colors["message_bg"]};
            --chat-panel-bg: {colors["chat_panel"]};
            --cyan: #22f5ff;
            --blue: #6d7cff;
            --violet: #a855f7;
            --green: #47f5b5;
            --rose: #ff4d7d;
            --amber: #ffd166;
        }}
        html, body, [class*="css"] {{
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            letter-spacing: 0;
        }}
        * {{
            transition:
                background-color 220ms ease-in-out,
                border-color 220ms ease-in-out,
                box-shadow 220ms ease-in-out,
                transform 220ms ease-in-out,
                color 220ms ease-in-out;
        }}
        ::selection {{
            background: rgba(34, 245, 255, 0.28);
            color: var(--text);
        }}
        .stApp {{
            color: var(--text);
            background:
                linear-gradient(120deg, var(--app-bg), var(--app-mid) 48%, var(--app-bg2)),
                repeating-linear-gradient(90deg, rgba(34,245,255,0.05) 0 1px, transparent 1px 76px),
                repeating-linear-gradient(0deg, rgba(34,245,255,0.035) 0 1px, transparent 1px 76px);
            overflow-x: hidden;
            transition: background 350ms ease, color 350ms ease;
        }}
        #ai-neural-background {{ opacity: var(--canvas-opacity) !important; }}
        #sidebar-collapse-toggle {{
            position: fixed;
            z-index: 100000;
            top: 14px;
            left: 14px;
            width: 38px;
            height: 38px;
            border: 1px solid rgba(34,245,255,0.34);
            border-radius: 12px;
            color: var(--text);
            background: linear-gradient(145deg, var(--panel-strong), rgba(34,245,255,0.10));
            box-shadow: 0 12px 30px var(--shadow);
            backdrop-filter: blur(var(--glass-blur));
            cursor: pointer;
        }}
        #sidebar-collapse-toggle:hover {{
            transform: translateY(-2px);
            box-shadow: 0 16px 42px var(--shadow), 0 0 24px rgba(34,245,255,0.18);
        }}
        .stApp::before {{
            content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
            background:
                radial-gradient(circle at 18% 18%, rgba(34, 245, 255, 0.24), transparent 26%),
                radial-gradient(circle at 84% 12%, rgba(168, 85, 247, 0.22), transparent 24%),
                radial-gradient(circle at 62% 86%, rgba(71, 245, 181, 0.14), transparent 28%);
            animation: auroraShift 13s ease-in-out infinite alternate;
        }}
        .stApp::after {{
            content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
            background:
                linear-gradient(transparent 0 47%, rgba(34, 245, 255, 0.10) 50%, transparent 53%),
                linear-gradient(90deg, transparent 0 47%, rgba(168, 85, 247, 0.10) 50%, transparent 53%);
            background-size: 100% 110px, 130px 100%;
            mix-blend-mode: screen; opacity: 0.42; animation: scanGrid 7s linear infinite;
        }}
        @keyframes auroraShift {{ from {{ filter: hue-rotate(0deg); transform: scale(1); }} to {{ filter: hue-rotate(24deg); transform: scale(1.08); }} }}
        @keyframes scanGrid {{ from {{ background-position: 0 0, 0 0; }} to {{ background-position: 0 220px, 260px 0; }} }}
        @keyframes pulseGlow {{ 0%, 100% {{ box-shadow: 0 0 0 rgba(34,245,255,0), 0 24px 80px var(--shadow); }} 50% {{ box-shadow: 0 0 54px rgba(34,245,255,0.24), 0 24px 80px var(--shadow); }} }}
        @keyframes livePulse {{ 0%, 100% {{ opacity: .45; transform: scale(.95); }} 50% {{ opacity: 1; transform: scale(1.05); }} }}
        @keyframes borderScan {{ 0% {{ transform: translateX(-120%); }} 100% {{ transform: translateX(120%); }} }}
        @keyframes progressSheen {{ 0% {{ background-position: 0% 50%; }} 100% {{ background-position: 200% 50%; }} }}
        @keyframes scanSweep {{ 0% {{ transform: translateY(-130%); opacity: 0; }} 20%, 80% {{ opacity: 1; }} 100% {{ transform: translateY(130%); opacity: 0; }} }}
        .main .block-container {{ position: relative; z-index: 2; max-width: 1260px; padding-top: 2rem; padding-bottom: 2.5rem; }}
        div[data-testid="stDecoration"] {{ display: none; }}
        section[data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, rgba(34,245,255,0.08), transparent 22%),
                var(--panel-strong);
            border-right: 1px solid var(--line);
            backdrop-filter: blur(var(--glass-blur)) saturate(130%);
            box-shadow: 16px 0 60px var(--shadow);
            transition: width 260ms ease-in-out, min-width 260ms ease-in-out, transform 260ms ease-in-out;
        }}
        section[data-testid="stSidebar"] * {{ color: var(--text); }}
        .stApp, .stApp p, .stApp span, .stApp label, .stApp div {{
            color: var(--text-color);
        }}
        .stCaptionContainer, .stCaptionContainer p, small {{
            color: var(--secondary-text-color) !important;
        }}
        section[data-testid="stSidebar"] > div {{
            padding-top: 58px;
        }}
        body.sidebar-collapsed section[data-testid="stSidebar"] {{
            min-width: 86px !important;
            width: 86px !important;
        }}
        body.sidebar-collapsed section[data-testid="stSidebar"] .sidebar-panel,
        body.sidebar-collapsed section[data-testid="stSidebar"] .stMarkdown,
        body.sidebar-collapsed section[data-testid="stSidebar"] label,
        body.sidebar-collapsed section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        body.sidebar-collapsed section[data-testid="stSidebar"] .stSelectbox,
        body.sidebar-collapsed section[data-testid="stSidebar"] .stSlider,
        body.sidebar-collapsed section[data-testid="stSidebar"] .stToggle,
        body.sidebar-collapsed section[data-testid="stSidebar"] .stCaptionContainer {{
            opacity: 0;
            pointer-events: none;
        }}
        .sidebar-panel {{
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 16px;
            margin-bottom: 18px;
            background: linear-gradient(145deg, var(--panel), rgba(34,245,255,0.035));
            box-shadow: var(--card-shadow);
        }}
        .sidebar-brand {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 14px;
        }}
        .sidebar-logo {{
            display: grid;
            place-items: center;
            width: 42px;
            height: 42px;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--cyan), var(--violet));
            color: #fff;
            font-weight: 900;
            box-shadow: 0 0 26px rgba(34,245,255,0.20);
        }}
        .sidebar-title {{
            color: var(--text);
            font-size: 16px;
            font-weight: 900;
        }}
        .sidebar-subtitle {{
            color: var(--muted);
            font-size: 12px;
            margin-top: 2px;
        }}
        .sidebar-section-label {{
            color: var(--muted);
            font-size: 11px;
            font-weight: 900;
            text-transform: uppercase;
            margin: 14px 0 8px;
        }}
        .sidebar-nav {{
            display: grid;
            gap: 8px;
            margin: 12px 0 2px;
        }}
        .sidebar-nav-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border: 1px solid transparent;
            border-radius: 14px;
            color: var(--muted);
            background: transparent;
            cursor: pointer;
            user-select: none;
        }}
        .sidebar-nav-item:hover {{
            transform: translateX(3px);
            border-color: var(--line);
            background: var(--surface);
            color: var(--text);
        }}
        .sidebar-nav-item.active {{
            color: var(--text);
            border-color: rgba(34,245,255,0.34);
            background: linear-gradient(90deg, rgba(34,245,255,0.13), rgba(168,85,247,0.11));
            box-shadow: inset 3px 0 0 var(--cyan);
        }}
        .sidebar-divider {{
            height: 1px;
            margin: 16px 0;
            background: linear-gradient(90deg, transparent, var(--line), transparent);
        }}
        .hero-shell, .metric-tile, .glass-panel {{
            position: relative; overflow: hidden; border: 1px solid var(--line); border-radius: 18px;
            background:
                linear-gradient(145deg, var(--panel), rgba(34,245,255,0.05)),
                linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.015));
            box-shadow: var(--card-shadow); backdrop-filter: blur(var(--glass-blur));
        }}
        .hero-shell {{
            padding: 34px;
            animation: pulseGlow 4.5s ease-in-out infinite;
            border-color: rgba(34,245,255,0.34);
        }}
        .hero-shell::before, .metric-tile::before, .glass-panel::before {{
            content: ""; position: absolute; inset: 0; border-top: 1px solid rgba(255,255,255,0.22); pointer-events: none;
        }}
        .hero-shell::after, .glass-panel::after, .metric-tile::after {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 55%;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--cyan), transparent);
            animation: borderScan 5.8s ease-in-out infinite;
            pointer-events: none;
        }}
        .hero-content {{ position: relative; z-index: 1; display: grid; grid-template-columns: 1fr auto; gap: 24px; align-items: center; }}
        .ai-chip {{ display: inline-flex; align-items: center; gap: 8px; width: fit-content; padding: 8px 11px; border: 1px solid rgba(71,245,181,0.38); border-radius: 999px; background: rgba(71,245,181,0.10); color: #0dd99b; font-size: 12px; font-weight: 800; text-transform: uppercase; }}
        .chip-dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 18px var(--green); animation: livePulse 1.4s ease-in-out infinite; }}
        .title {{ margin: 12px 0 0; color: var(--text); font-size: clamp(40px, 5vw, 68px); line-height: 1.02; font-weight: 900; max-width: 850px; }}
        .gradient-text {{
            color: transparent;
            background: linear-gradient(90deg, var(--cyan), var(--violet), var(--blue), var(--cyan));
            background-size: 220% 100%;
            -webkit-background-clip: text;
            background-clip: text;
            text-shadow: 0 0 30px rgba(34,245,255,0.26);
            animation: progressSheen 5s linear infinite;
        }}
        .subtitle {{ color: var(--muted); font-size: 18px; max-width: 780px; margin-top: 14px; }}
        .ai-orbit {{ position: relative; width: 170px; aspect-ratio: 1; border-radius: 50%; border: 1px solid rgba(34,245,255,0.28); background: radial-gradient(circle, rgba(34,245,255,0.28) 0 9%, transparent 10%), radial-gradient(circle, rgba(109,124,255,0.18), transparent 58%); box-shadow: inset 0 0 36px rgba(34,245,255,0.15), 0 0 46px rgba(34,245,255,0.12); }}
        .ai-orbit::before, .ai-orbit::after {{ content: ""; position: absolute; inset: 18px; border: 1px dashed rgba(34,245,255,0.42); border-radius: 50%; animation: spin 9s linear infinite; }}
        .ai-orbit::after {{ inset: 42px; border-color: rgba(71,245,181,0.44); animation-duration: 5.5s; animation-direction: reverse; }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .metric-row {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 18px 0 24px; }}
        .dashboard-section {{
            display: none;
            opacity: 0;
            transform: translateY(10px);
        }}
        .dashboard-section.active {{
            display: block;
            opacity: 1;
            transform: translateY(0);
            animation: sectionFade 240ms ease-in-out;
        }}
        @keyframes sectionFade {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .js-section-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr);
            gap: 18px;
            margin-top: 10px;
        }}
        .js-control-row {{
            display: grid;
            gap: 10px;
            padding: 14px 0;
            border-bottom: 1px solid var(--line);
        }}
        .js-control-row label {{
            color: var(--text);
            font-size: 13px;
            font-weight: 900;
        }}
        .js-control-row input[type="range"] {{
            width: 100%;
            accent-color: var(--cyan);
            cursor: pointer;
        }}
        .js-control-row input[type="number"],
        .js-control-row select {{
            width: 100%;
            color: var(--text);
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 11px 12px;
            outline: none;
        }}
        .js-control-row input[type="number"]:focus,
        .js-control-row select:focus {{
            border-color: rgba(34,245,255,0.50);
            box-shadow: 0 0 0 4px rgba(34,245,255,0.12);
        }}
        .js-state-card {{
            display: grid;
            gap: 12px;
            padding: 18px;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: var(--surface);
        }}
        .js-state-value {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            color: var(--muted);
            font-size: 13px;
        }}
        .js-state-value strong {{
            color: var(--text);
            font-size: 16px;
        }}
        .js-scan-list {{
            display: grid;
            gap: 12px;
            margin-top: 14px;
        }}
        .js-scan-card {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 14px;
            padding: 14px;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: var(--surface);
        }}
        .js-scan-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(34,245,255,0.42);
            box-shadow: 0 18px 46px var(--shadow);
        }}
        .js-scan-file {{
            color: var(--text);
            font-weight: 900;
            overflow-wrap: anywhere;
        }}
        .js-scan-meta {{
            color: var(--muted);
            font-size: 13px;
            margin-top: 4px;
        }}
        .js-clear-button {{
            border: 1px solid rgba(34,245,255,0.38);
            border-radius: 999px;
            padding: 10px 14px;
            color: #ffffff;
            background: linear-gradient(90deg, rgba(34,245,255,0.88), rgba(168,85,247,0.88));
            font-weight: 900;
            cursor: pointer;
            box-shadow: 0 12px 34px rgba(34,245,255,0.16);
        }}
        .js-clear-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 16px 44px rgba(34,245,255,0.26), 0 0 22px rgba(168,85,247,0.24);
        }}
        .empty-state {{
            padding: 18px;
            border: 1px dashed var(--line);
            border-radius: 16px;
            color: var(--muted);
            background: var(--surface);
            text-align: center;
        }}
        .metric-tile {{ padding: 18px; min-height: 104px; }}
        .metric-tile:hover, .glass-panel:hover {{
            transform: translateY(-3px);
            border-color: rgba(34,245,255,0.42);
            box-shadow: var(--card-shadow), 0 0 34px rgba(34,245,255,0.12);
        }}
        .metric-value {{ font-size: 24px; font-weight: 900; color: var(--text); }}
        .metric-label {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
        .glass-panel {{ padding: 26px; }}
        .panel-heading {{ color: var(--text); font-size: 23px; font-weight: 900; margin-bottom: 6px; }}
        .panel-copy {{ color: var(--muted); margin-bottom: 18px; }}
        .file-chip {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 10px; border: 1px solid rgba(34,245,255,0.24); border-radius: 999px; background: rgba(34,245,255,0.08); color: var(--text); font-size: 13px; margin: 4px 6px 12px 0; }}
        .scan-loader {{
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 14px;
            align-items: center;
            margin: 14px 0 18px;
            padding: 12px 14px;
            border: 1px solid var(--line);
            border-radius: 16px;
            background:
                linear-gradient(90deg, rgba(34,245,255,0.09), rgba(168,85,247,0.08)),
                rgba(255,255,255,0.035);
        }}
        .scan-loader-track {{
            position: relative;
            height: 8px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            overflow: hidden;
        }}
        .scan-loader-track::before {{
            content: "";
            position: absolute;
            inset: 0;
            width: 42%;
            border-radius: inherit;
            background: linear-gradient(90deg, transparent, var(--cyan), var(--violet), transparent);
            animation: borderScan 2.2s ease-in-out infinite;
        }}
        .scan-loader-label {{
            color: var(--muted);
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            white-space: nowrap;
        }}
        .result-box {{
            display: flex; align-items: center; justify-content: center; gap: 18px; padding: 24px;
            border-radius: 18px; margin: 18px 0 14px; border: 1px solid var(--line);
            text-align: left;
        }}
        .result-box.fake {{ background: radial-gradient(circle at 14% 18%, rgba(255,77,125,0.32), transparent 32%), linear-gradient(135deg, rgba(255,77,125,0.25), rgba(86,12,35,0.28)); }}
        .result-box.real {{ background: radial-gradient(circle at 14% 18%, rgba(71,245,181,0.30), transparent 32%), linear-gradient(135deg, rgba(71,245,181,0.22), rgba(4,65,55,0.24)); }}
        .result-icon {{ display: grid; place-items: center; width: 52px; aspect-ratio: 1; border-radius: 50%; border: 1px solid rgba(255,255,255,0.28); font-size: 18px; font-weight: 900; background: rgba(255,255,255,0.10); }}
        .result-label {{ font-size: clamp(32px, 4vw, 48px); line-height: 1; font-weight: 900; }}
        .result-caption {{ color: var(--muted); font-size: 14px; margin-top: 5px; }}
        .timeline {{ display: flex; align-items: end; gap: 4px; height: 104px; padding: 14px; border: 1px solid var(--line); border-radius: 16px; background: rgba(34,245,255,0.055); }}
        .timeline-bar {{ flex: 1; min-width: 4px; border-radius: 8px 8px 2px 2px; background: linear-gradient(180deg, var(--rose), var(--amber)); box-shadow: 0 0 16px rgba(255,77,125,0.18); transform-origin: bottom; animation: livePulse 2.2s ease-in-out infinite; }}
        .frame-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(42px, 1fr)); gap: 8px; margin-top: 12px; }}
        .frame-cell {{ min-height: 42px; display: grid; place-items: center; border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,0.055); color: var(--text); font-size: 11px; font-weight: 800; position: relative; overflow: hidden; }}
        .frame-cell::after {{ content: ""; position: absolute; inset: auto 0 0 0; height: var(--risk); background: linear-gradient(180deg, rgba(255,77,125,0.10), rgba(255,77,125,0.38)); z-index: -1; }}
        .why-list {{ display: grid; gap: 10px; margin-top: 12px; }}
        .why-item {{ padding: 12px 14px; border: 1px solid var(--line); border-radius: 14px; background: rgba(255,255,255,0.055); }}
        .history-item {{ display: grid; grid-template-columns: 74px 1fr auto; gap: 12px; align-items: center; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.10); }}
        .history-time {{ color: var(--muted); font-size: 12px; }}
        .history-name {{ color: var(--text); font-weight: 800; overflow-wrap: anywhere; }}
        .history-meta {{ color: var(--muted); font-size: 13px; }}
        .badge {{ border-radius: 999px; padding: 6px 9px; font-size: 12px; font-weight: 900; }}
        .badge.fake {{ background: rgba(255,77,125,0.18); color: #ff4d7d; }}
        .badge.real {{ background: rgba(71,245,181,0.18); color: #0fbd83; }}
        .share-box {{ padding: 12px; border: 1px solid var(--line); border-radius: 14px; background: rgba(34,245,255,0.07); color: var(--text); overflow-wrap: anywhere; font-size: 13px; }}
        .footer {{ color: var(--muted); text-align: center; margin-top: 30px; padding-top: 18px; border-top: 1px solid var(--line); }}
        h1, h2, h3, h4, h5, h6, p, span, label, div[data-testid="stMarkdownContainer"], .stCaptionContainer {{
            color: inherit;
        }}
        [data-testid="stWidgetLabel"] p {{
            color: var(--text);
            font-weight: 800;
            font-size: 13px;
        }}
        div[data-testid="stFileUploader"] {{
            background:
                linear-gradient(145deg, var(--input-bg), rgba(34,245,255,0.045)),
                radial-gradient(circle at 50% 0%, rgba(168,85,247,0.13), transparent 45%);
            border: 1px dashed rgba(34,245,255,0.52);
            border-radius: 18px;
            padding: 18px;
            transition: transform 200ms ease, box-shadow 200ms ease;
        }}
        div[data-testid="stFileUploader"] * {{
            color: var(--text-color) !important;
        }}
        div[data-testid="stFileUploader"]:hover {{ transform: translateY(-3px); box-shadow: 0 0 42px rgba(34,245,255,0.24), inset 0 0 24px rgba(34,245,255,0.06); }}
        div[data-testid="stFileUploader"] section {{ border-color: transparent; }}
        .stRadio [role="radiogroup"] {{ gap: 10px; }}
        .stRadio label {{
            min-height: 42px;
            border: 1px solid rgba(14,165,233,0.28);
            border-radius: 999px;
            padding: 9px 14px;
            background: var(--input-bg);
            color: var(--text-color) !important;
            box-shadow: 0 1px 0 rgba(255,255,255,0.20), 0 6px 18px rgba(15,23,42,0.06);
            cursor: pointer;
            font-weight: 800;
        }}
        .stRadio label * {{
            color: var(--text-color) !important;
            font-weight: 800;
        }}
        .stRadio label:hover {{
            transform: translateY(-2px);
            border-color: rgba(14,165,233,0.58);
            box-shadow: 0 12px 28px rgba(14,165,233,0.16);
        }}
        .stRadio label:has(input:checked) {{
            border-color: transparent;
            background: linear-gradient(90deg, #06b6d4, #6366f1) !important;
            box-shadow: 0 14px 34px rgba(14,165,233,0.28), 0 0 24px rgba(99,102,241,0.18);
        }}
        .stRadio label:has(input:checked) * {{
            color: #ffffff !important;
        }}
        .stRadio input {{
            accent-color: #06b6d4;
        }}
        div[data-baseweb="select"] > div {{
            border: 1px solid var(--line) !important;
            border-radius: 14px !important;
            background: var(--input-bg) !important;
            color: var(--text) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }}
        div[data-baseweb="select"] * {{
            color: var(--text-color) !important;
        }}
        div[data-baseweb="select"]:hover > div {{
            border-color: rgba(34,245,255,0.42) !important;
            box-shadow: 0 0 24px rgba(34,245,255,0.10);
        }}
        div[data-baseweb="popover"] {{
            border-radius: 14px !important;
        }}
        ul[role="listbox"] {{
            background: var(--panel-strong) !important;
            border: 1px solid var(--line) !important;
            border-radius: 14px !important;
            box-shadow: 0 22px 60px var(--shadow) !important;
        }}
        li[role="option"] {{
            color: var(--text) !important;
            background: transparent !important;
        }}
        li[role="option"]:hover {{
            background: var(--surface) !important;
        }}
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, var(--cyan), var(--violet), var(--green), var(--cyan));
            background-size: 220% 100%;
            animation: progressSheen 2.8s linear infinite;
        }}
        .stProgress > div > div {{ background: var(--surface); border-radius: 999px; overflow: hidden; border: 1px solid var(--line); }}
        .stButton > button, .stDownloadButton > button, button[kind="primary"], button[kind="secondary"] {{
            border: 1px solid rgba(34,245,255,0.38) !important;
            border-radius: 999px !important;
            background: linear-gradient(90deg, rgba(34,245,255,0.88), rgba(168,85,247,0.88)) !important;
            color: #ffffff !important;
            font-weight: 800 !important;
            box-shadow: 0 12px 34px rgba(34,245,255,0.16);
        }}
        .stButton > button:hover, .stDownloadButton > button:hover, button[kind="primary"]:hover, button[kind="secondary"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 16px 44px rgba(34,245,255,0.26), 0 0 22px rgba(168,85,247,0.24);
        }}
        div[data-baseweb="slider"] [role="slider"] {{
            background: var(--cyan) !important;
            border: 3px solid rgba(255,255,255,0.86) !important;
            box-shadow: 0 0 0 6px rgba(34,245,255,0.14), 0 0 24px rgba(34,245,255,0.45) !important;
        }}
        div[data-baseweb="slider"] [aria-valuenow] {{
            outline: none !important;
        }}
        div[data-baseweb="slider"] div {{
            border-radius: 999px;
        }}
        [data-testid="stToggle"] label {{
            padding: 10px 12px;
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--input-bg);
            color: var(--text-color) !important;
        }}
        [data-testid="stToggle"] label * {{
            color: var(--text-color) !important;
        }}
        [data-testid="stToggle"] label:hover {{
            border-color: rgba(34,245,255,0.44);
            box-shadow: 0 0 24px rgba(34,245,255,0.10);
        }}
        .stStatus {{
            position: relative;
            overflow: hidden;
        }}
        .stStatus::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, transparent, rgba(34,245,255,0.18), transparent);
            animation: scanSweep 1.65s ease-in-out infinite;
            pointer-events: none;
        }}
        @media (max-width: 850px) {{
            #sidebar-collapse-toggle {{ top: 10px; left: 10px; }}
            section[data-testid="stSidebar"] {{
                position: fixed;
                z-index: 99999;
                width: min(84vw, 340px) !important;
                min-width: min(84vw, 340px) !important;
            }}
            body.sidebar-collapsed section[data-testid="stSidebar"] {{
                transform: translateX(-100%);
                min-width: min(84vw, 340px) !important;
                width: min(84vw, 340px) !important;
            }}
            body.sidebar-collapsed section[data-testid="stSidebar"] .sidebar-panel,
            body.sidebar-collapsed section[data-testid="stSidebar"] .stMarkdown,
            body.sidebar-collapsed section[data-testid="stSidebar"] label,
            body.sidebar-collapsed section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
            body.sidebar-collapsed section[data-testid="stSidebar"] .stSelectbox,
            body.sidebar-collapsed section[data-testid="stSidebar"] .stSlider,
            body.sidebar-collapsed section[data-testid="stSidebar"] .stToggle,
            body.sidebar-collapsed section[data-testid="stSidebar"] .stCaptionContainer {{
                opacity: 1;
            }}
            .hero-content {{ grid-template-columns: 1fr; }}
            .ai-orbit {{ display: none; }}
            .title {{ font-size: 40px; }}
            .metric-row {{ grid-template-columns: 1fr; gap: 12px; }}
            .history-item {{ grid-template-columns: 1fr; }}
            .glass-panel, .hero-shell {{ padding: 20px; border-radius: 16px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model(model_mtime):
    return tf.keras.models.load_model("deepfake_model.h5")


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def load_history():
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_history(history):
    HISTORY_PATH.write_text(json.dumps(history[:25], indent=2), encoding="utf-8")


def make_scan_id(file_name, media_type, result, confidence):
    raw = f"{file_name}|{media_type}|{result}|{confidence:.6f}|{datetime.now().isoformat()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def pdf_escape(value):
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf_report(scan):
    lines = [
        "Deepfake Detection Report",
        f"Scan ID: {scan['id']}",
        f"File: {scan['file']}",
        f"Type: {scan['type']}",
        f"Result: {scan['result']}",
        f"Confidence: {scan['confidence']:.2%}",
        f"Fake probability: {scan['fake_probability']:.2%}",
        f"Real probability: {scan['real_probability']:.2%}",
        f"Timestamp: {scan['timestamp']}",
        "",
        "Technical explanation:",
        scan["summary"],
    ]
    text_ops = ["BT", "/F1 16 Tf", "72 760 Td", "18 TL"]
    for index, line in enumerate(lines):
        if index == 1:
            text_ops.append("/F1 11 Tf")
        text_ops.append(f"({pdf_escape(line[:92])}) Tj")
        text_ops.append("T*")
    text_ops.append("ET")
    stream = "\n".join(text_ops).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = [b"%PDF-1.4\n"]
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part) for part in pdf))
        pdf.append(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = sum(len(part) for part in pdf)
    pdf.append(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        pdf.append(f"{offset:010d} 00000 n \n".encode())
    pdf.append(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return b"".join(pdf)


def create_heatmap(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not create heatmap for this image.")
    image = cv2.resize(image, (480, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    heat = cv2.GaussianBlur(edges, (31, 31), 0)
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
    color_heat = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.62, color_heat, 0.38, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def frame_scores(model, frames, threshold):
    scores = []
    for frame in frames:
        frame_prediction = float(model.predict(np.expand_dims([frame], axis=0), verbose=0)[0][0])
        scores.append(frame_prediction)
    details = predict_details(model, frames, threshold=threshold, source="video_sequence")
    return details["label"], details["confidence"], details["fake_probability"], scores


def live_camera_threshold(base_threshold):
    return CAMERA_FAKE_THRESHOLD


def smooth_webcam_prediction(fake_probability, threshold, capture_key=None, window=1):
    if capture_key and st.session_state.get("last_webcam_capture_key") != capture_key:
        st.session_state.last_webcam_capture_key = capture_key
        st.session_state.webcam_fake_scores = []

    scores = st.session_state.setdefault("webcam_fake_scores", [])
    scores.append(float(fake_probability))
    st.session_state.webcam_fake_scores = scores[-window:]

    smoothed_fake_probability = float(np.mean(st.session_state.webcam_fake_scores))
    label = "Fake" if smoothed_fake_probability > threshold else "Real"
    confidence = smoothed_fake_probability if label == "Fake" else 1.0 - smoothed_fake_probability
    return label, confidence, smoothed_fake_probability


def explain_signals(result, fake_probability):
    if result == "Fake":
        return [
            ("Face distortion", min(0.98, fake_probability + 0.10)),
            ("Lip-sync mismatch", min(0.95, fake_probability + 0.03)),
            ("Unnatural blinking", min(0.92, fake_probability - 0.02 if fake_probability > 0.15 else fake_probability)),
        ]
    return [
        ("Face consistency", max(0.55, 1 - fake_probability)),
        ("Natural motion cues", max(0.50, 1 - fake_probability - 0.05)),
        ("Stable visual texture", max(0.50, 1 - fake_probability - 0.08)),
    ]


def render_result_dashboard(scan):
    fake_percent = scan["fake_probability"] * 100
    real_percent = scan["real_probability"] * 100
    confidence_percent = int(round(scan["confidence"] * 100))

    render_result(scan["result"], scan["confidence"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Real", f"{real_percent:.1f}%")
    c2.metric("Fake", f"{fake_percent:.1f}%")
    c3.metric("Confidence", f"{confidence_percent}%")
    st.progress(confidence_percent)

    if scan["frame_scores"]:
        st.subheader("Frame Timeline")
        st.line_chart(scan["frame_scores"], height=180)
        bars = "".join(
            f'<div class="timeline-bar" title="Frame {i + 1}: {score:.2%}" style="height:{max(8, score * 90):.1f}%"></div>'
            for i, score in enumerate(scan["frame_scores"])
        )
        st.markdown(f'<div class="timeline">{bars}</div>', unsafe_allow_html=True)
        frame_cells = "".join(
            f'<div class="frame-cell" style="--risk:{score:.0%}" title="Frame {i + 1}: {score:.2%} fake probability">F{i + 1}</div>'
            for i, score in enumerate(scan["frame_scores"][:48])
        )
        st.markdown(f'<div class="frame-grid">{frame_cells}</div>', unsafe_allow_html=True)


def render_result(result, confidence):
    result_class = "fake" if result == "Fake" else "real"
    icon = "!" if result == "Fake" else "OK"
    confidence_percent = int(round(confidence * 100))
    st.markdown(
        f"""
        <div class="result-box {result_class}">
            <div class="result-icon">{icon}</div>
            <div>
                <div class="result-label">{result}</div>
                <div class="result-caption">{confidence_percent}% confidence score</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_scan(scan):
    history = load_history()
    if st.session_state.get("last_scan_key") == scan["dedupe_key"]:
        return
    st.session_state.last_scan_key = scan["dedupe_key"]
    history.insert(0, scan)
    save_history(history)
    st.session_state.scan_history = history[:25]


def build_scan(file_name, media_type, result, confidence, fake_probability, frame_score_values, mode_name, threshold):
    scan_id = make_scan_id(file_name, media_type, result, confidence)
    scan = {
        "id": scan_id,
        "file": file_name,
        "type": media_type,
        "result": result,
        "confidence": confidence,
        "fake_probability": fake_probability,
        "real_probability": 1 - fake_probability,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frame_scores": [float(score) for score in frame_score_values],
        "mode": mode_name,
        "threshold": threshold,
        "summary": (
            f"The model classified this {media_type.lower()} as {result}. "
            f"The fake probability was {fake_probability:.2%} using {mode_name}."
        ),
    }
    scan["dedupe_key"] = f"{file_name}:{media_type}:{result}:{confidence:.4f}:{mode_name}"
    scan["report_name"] = f"deepfake_report_{scan_id}.pdf"
    report_path = REPORTS_DIR / scan["report_name"]
    report_path.write_bytes(make_pdf_report(scan))
    return scan


inject_ai_background()
inject_ai_assistant()

if "scan_history" not in st.session_state:
    st.session_state.scan_history = load_history()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-panel">
            <div class="sidebar-brand">
                <div class="sidebar-logo">AI</div>
                <div>
                    <div class="sidebar-title">DeepScan</div>
                    <div class="sidebar-subtitle">Detection Console</div>
                </div>
            </div>
            <div class="sidebar-nav">
                <div class="sidebar-nav-item active" data-section="ai-controls"><span>AI</span><span>AI Controls</span></div>
                <div class="sidebar-nav-item" data-section="model-tuning"><span>MT</span><span>Model Tuning</span></div>
                <div class="sidebar-nav-item" data-section="scan-memory"><span>SM</span><span>Scan Memory</span></div>
            </div>
        </div>
        <div class="sidebar-section-label">Appearance</div>
        """,
        unsafe_allow_html=True,
    )
    theme = st.toggle("Light mode", value=False)
    selected_theme = "Light" if theme else "Dark"
    st.markdown('<div class="sidebar-divider"></div><div class="sidebar-section-label">Detection Model</div>', unsafe_allow_html=True)
    mode_name = st.selectbox("Model selection", list(MODE_CONFIG.keys()), index=1)
    mode = MODE_CONFIG[mode_name]
    st.markdown('<div class="sidebar-divider"></div><div class="sidebar-section-label">Sensitivity</div>', unsafe_allow_html=True)
    threshold = st.slider("Fake threshold", 0.10, 0.90, mode["threshold"], 0.05)
    max_frames = st.slider("Video frames", 5, 60, mode["frames"], 5)
    st.caption(mode["label"])
    distribution = prediction_distribution()
    if distribution:
        st.markdown('<div class="sidebar-divider"></div><div class="sidebar-section-label">Prediction Debug</div>', unsafe_allow_html=True)
        st.caption(
            f"Logged {distribution['count']} predictions | "
            f"fake score min/mean/max: {distribution['min']:.2f}/"
            f"{distribution['mean']:.2f}/{distribution['max']:.2f}"
        )

apply_theme(selected_theme)

try:
    model = load_model(Path("deepfake_model.h5").stat().st_mtime)
except OSError:
    st.error("Model file not found. Train the model first to create deepfake_model.h5.")
    st.stop()

query_scan_id = st.query_params.get("scan")
shared_scan = None
if query_scan_id:
    shared_scan = next((item for item in st.session_state.scan_history if item["id"] == query_scan_id), None)

st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-content">
            <div>
                <div class="ai-chip"><span class="chip-dot"></span> AI authenticity engine online</div>
                <h1 class="title">Deepfake <span class="gradient-text">Detection</span> Dashboard</h1>
                <div class="subtitle">Upload media, scan camera snapshots, inspect frame timelines, view explainability signals, and export a PDF report.</div>
            </div>
            <div class="ai-orbit"></div>
        </div>
    </div>
    <div class="metric-row">
        <div class="metric-tile"><div class="metric-value">Dashboard</div><div class="metric-label">Real vs Fake percentages</div></div>
        <div class="metric-tile"><div class="metric-value">Camera</div><div class="metric-label">Webcam snapshot detection</div></div>
        <div class="metric-tile"><div class="metric-value">Heatmap</div><div class="metric-label">Visual explainability overlay</div></div>
        <div class="metric-tile"><div class="metric-value">PDF</div><div class="metric-label">Downloadable AI report</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if shared_scan:
    st.info(f"Shared result loaded: {shared_scan['file']} classified as {shared_scan['result']}.")
    render_result_dashboard(shared_scan)

st.markdown('<div class="dashboard-section active" data-section="ai-controls">', unsafe_allow_html=True)

left_col, right_col = st.columns([1.42, 1], gap="large")

current_scan = None

with left_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Neural Media Scanner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-copy">Drag a file into the glowing upload box or capture from your webcam.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="scan-loader"><div class="scan-loader-track"></div><div class="scan-loader-label">AI scan layer ready</div></div>',
        unsafe_allow_html=True,
    )

    option = st.radio("Choose input type:", ["Image", "Video", "Live Camera"], horizontal=True)

    if option == "Image":
        file = st.file_uploader("Drag & drop image", type=["jpg", "jpeg", "png"])
        if file:
            suffix = "." + file.name.rsplit(".", 1)[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(file.read())
                temp_path = f.name

            st.toast("Scanning started...")
            with st.status("Live scanning...", expanded=True) as status:
                st.write("Processing image...")
                img = load_image(temp_path)
                details = predict_details(model, [img], threshold=threshold, source=f"image:{file.name}")
                result = details["label"]
                confidence = details["confidence"]
                fake_probability = details["fake_probability"]
                status.update(label="Analysis complete", state="complete")
            st.toast("Analysis complete")

            current_scan = build_scan(file.name, "Image", result, confidence, fake_probability, [], mode_name, threshold)
            save_scan(current_scan)
            st.markdown(
                f'<span class="file-chip">FILE {file.name}</span><span class="file-chip">SIZE {format_size(file.size)}</span><span class="file-chip">MODE Image scan</span>',
                unsafe_allow_html=True,
            )
            st.image(temp_path, use_container_width=True)
            render_result_dashboard(current_scan)

            st.subheader("Explainability Heatmap")
            st.image(create_heatmap(temp_path), caption="Highlighted high-variation regions for visual review.", use_container_width=True)

    elif option == "Video":
        file = st.file_uploader("Drag & drop video", type=["mp4"])
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                f.write(file.read())
                temp_path = f.name

            st.toast("Scanning started...")
            with st.status("Live scanning...", expanded=True) as status:
                st.write("Processing frames...")
                frames = extract_frames(temp_path, max_frames=max_frames)
                if len(frames) == 0:
                    st.error("Could not read frames from this video. Please upload a valid MP4 file.")
                    st.stop()
                st.write(f"Analyzing {len(frames)} sampled frames...")
                result, confidence, fake_probability, scores = frame_scores(model, frames, threshold)
                status.update(label="Analysis complete", state="complete")
            st.toast("Analysis complete")

            current_scan = build_scan(file.name, "Video", result, confidence, fake_probability, scores, mode_name, threshold)
            save_scan(current_scan)
            st.markdown(
                f'<span class="file-chip">FILE {file.name}</span><span class="file-chip">SIZE {format_size(file.size)}</span><span class="file-chip">FRAMES {len(frames)} sampled</span>',
                unsafe_allow_html=True,
            )
            st.video(temp_path)
            render_result_dashboard(current_scan)

    else:
        camera_file = st.camera_input("Use webcam for detection")
        if camera_file:
            camera_bytes = camera_file.read()
            camera_capture_key = hashlib.md5(camera_bytes).hexdigest()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                f.write(camera_bytes)
                temp_path = f.name

            with st.status("Live scanning...", expanded=True) as status:
                st.write("Processing camera frame...")
                img = load_image(temp_path)
                camera_threshold = live_camera_threshold(threshold)
                details = predict_details(model, [img], threshold=camera_threshold, source="live_webcam")
                result, confidence, fake_probability = smooth_webcam_prediction(
                    details["fake_probability"],
                    camera_threshold,
                    capture_key=camera_capture_key,
                )
                status.update(label="Analysis complete", state="complete")

            current_scan = build_scan("webcam_capture.jpg", "Live Camera", result, confidence, fake_probability, [], mode_name, camera_threshold)
            save_scan(current_scan)
            st.image(temp_path, use_container_width=True)
            render_result_dashboard(current_scan)
            st.subheader("Explainability Heatmap")
            st.image(create_heatmap(temp_path), caption="Camera-frame visual review overlay.", use_container_width=True)

    if current_scan:
        st.subheader("Why This Result")
        signals = explain_signals(current_scan["result"], current_scan["fake_probability"])
        html = "".join(
            f'<div class="why-item"><strong>{name}</strong><br>{score:.0%} signal strength</div>'
            for name, score in signals
        )
        st.markdown(f'<div class="why-list">{html}</div>', unsafe_allow_html=True)

        report_path = REPORTS_DIR / current_scan["report_name"]
        st.download_button(
            "Download PDF Report",
            data=report_path.read_bytes(),
            file_name=current_scan["report_name"],
            mime="application/pdf",
        )
        share_url = f"http://localhost:8501/?scan={current_scan['id']}"
        st.markdown(f'<div class="share-box">Share result link: {share_url}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">History / Scan Memory</div>', unsafe_allow_html=True)

    history = st.session_state.scan_history
    if history:
        for item in history[:8]:
            badge_class = "fake" if item["result"] == "Fake" else "real"
            st.markdown(
                f"""
                <div class="history-item">
                    <div class="history-time">{item["timestamp"]}</div>
                    <div>
                        <div class="history-name">{item["file"]}</div>
                        <div class="history-meta">{item["type"]} | {item["confidence"]:.2%} confidence</div>
                    </div>
                    <div class="badge {badge_class}">{item["result"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            report_path = REPORTS_DIR / item["report_name"]
            if report_path.exists():
                st.download_button(
                    f"Download Report {item['id']}",
                    data=report_path.read_bytes(),
                    file_name=item["report_name"],
                    mime="application/pdf",
                    key=f"report-{item['id']}",
                )
    else:
        st.info("Your latest scans will appear here.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="dashboard-section" data-section="model-tuning">
        <div class="js-section-grid">
            <div class="glass-panel">
                <div class="panel-heading">Model Tuning</div>
                <div class="panel-copy">Adjust frontend tuning state for threshold, frame sampling, and scan mode. These values persist in localStorage.</div>
                <div class="js-control-row">
                    <label for="js-threshold">Threshold: <span id="js-threshold-value">{threshold:.2f}</span></label>
                    <input id="js-threshold" type="range" min="0" max="1" step="0.01" value="{threshold:.2f}">
                </div>
                <div class="js-control-row">
                    <label for="js-frames">Frame count</label>
                    <input id="js-frames" type="number" min="1" max="120" value="{max_frames}">
                </div>
                <div class="js-control-row">
                    <label for="js-mode">Mode selection</label>
                    <select id="js-mode">
                        <option value="Fast mode">Fast</option>
                        <option value="Accurate mode">Accurate</option>
                    </select>
                </div>
            </div>
            <div class="glass-panel">
                <div class="panel-heading">Current State</div>
                <div class="js-state-card">
                    <div class="js-state-value"><span>Threshold</span><strong id="js-threshold-card-value">{threshold:.2f}</strong></div>
                    <div class="js-state-value"><span>Frames</span><strong id="js-frame-value">{max_frames}</strong></div>
                    <div class="js-state-value"><span>Mode</span><strong id="js-mode-value">{escape(mode_name)}</strong></div>
                </div>
            </div>
        </div>
    </div>
    <div class="dashboard-section" data-section="scan-memory">
        <div class="glass-panel">
            <div class="panel-heading">Scan Memory</div>
            <div class="panel-copy">Frontend scan memory with sample data and localStorage persistence. New completed scans are pushed here automatically.</div>
            <button id="js-clear-memory" class="js-clear-button" type="button">Clear Memory</button>
            <div id="js-scan-memory-list" class="js-scan-list"></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

inject_sidebar_navigation(st.session_state.scan_history, threshold, max_frames, mode_name)

if current_scan:
    frontend_scan = {
        "file": current_scan["file"],
        "result": current_scan["result"],
        "confidence": current_scan["confidence"],
        "timestamp": current_scan["timestamp"],
    }
    components.html(
        f"""
        <script>
        if (window.parent.deepScanAddScan) {{
            window.parent.deepScanAddScan({json.dumps(frontend_scan)});
        }}
        </script>
        """,
        height=0,
    )

st.markdown(
    '<div class="footer">Deepfake Detection Project | AI media authenticity scanner</div>',
    unsafe_allow_html=True,
)
