"""
Badminton AI dashboard.   Run:  .venv/Scripts/streamlit run app.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from badminton.court import (ALL_ZONES, COURT_LENGTH, COURT_WIDTH, NET_Y, SIDE_ZONES, load_corners,
                             save_corners, video_id_from_path, zone_center_m, zone_rect_m)
from badminton.predictor import MODEL_DIR

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_VIDEO = PROJECT_ROOT / "sample_dataset" / "00467.mp4"
UPLOAD_DIR = PROJECT_ROOT / "runs" / "uploads"
ANALYSIS_DIR = PROJECT_ROOT / "runs" / "analysis"
CORNER_NAMES = ["far-left", "far-right", "near-right", "near-left"]
PLAYER_COLORS = {1: "#E4572E", 2: "#2E86AB"}

st.set_page_config(page_title="Badminton AI", page_icon="🏸", layout="wide")


# ---------- helpers ----------
def first_frame(video: Path):
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def load_summary(video_id: str):
    path = ANALYSIS_DIR / video_id / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_metrics():
    path = MODEL_DIR / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def court_figure(values: dict = None, title: str = "", colorscale="Oranges", fmt="{:.0f}",
                 outline: dict = None, height=520):
    """Top-down court (far side at the top) with the 12 zones coloured by `values`."""
    values = values or {}
    vmax = max(values.values(), default=0) or 1
    fig = go.Figure()
    for zone in ALL_ZONES:
        x0, y0, x1, y1 = zone_rect_m(zone)
        v = values.get(zone, 0)
        color = px.colors.sample_colorscale(colorscale, [0.08 + 0.85 * v / vmax])[0] if v else "#2f6f4f"
        line = (outline or {}).get(zone, dict(color="white", width=1))
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=color, line=line, layer="below")
        cx, cy = zone_center_m(zone)
        text = f"<b>{zone}</b><br>{fmt.format(v)}" if v else f"<b>{zone}</b>"
        fig.add_annotation(x=cx, y=cy, text=text, showarrow=False, font=dict(color="white", size=13))
    fig.add_shape(type="line", x0=-0.3, y0=NET_Y, x1=COURT_WIDTH + 0.3, y1=NET_Y, line=dict(color="#FFD23F", width=4))
    fig.add_annotation(x=COURT_WIDTH + 0.35, y=NET_Y, text="net", showarrow=False, xanchor="left",
                       font=dict(color="#FFD23F"))
    fig.add_annotation(x=COURT_WIDTH / 2, y=-0.45, text="Player 1 (far side)", showarrow=False)
    fig.add_annotation(x=COURT_WIDTH / 2, y=COURT_LENGTH + 0.45, text="Player 2 (near side)", showarrow=False)
    fig.update_xaxes(range=[-0.6, COURT_WIDTH + 1.0], visible=False)
    fig.update_yaxes(range=[COURT_LENGTH + 0.8, -0.8], visible=False)
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=height, width=int(height * 0.62), margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


# ---------- sidebar: video + court corners ----------
st.sidebar.title("🏸 Badminton AI")
source = st.sidebar.radio("Video", ["Sample rally (00467)", "Upload a video"])
video_path = None
if source.startswith("Sample"):
    video_path = SAMPLE_VIDEO
else:
    upload = st.sidebar.file_uploader("Rally video", type=["mp4", "avi", "mov", "mkv"])
    if upload is not None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        video_path = UPLOAD_DIR / upload.name
        if not video_path.is_file() or video_path.stat().st_size != upload.size:
            video_path.write_bytes(upload.getbuffer())

if video_path is None:
    st.info("Upload a broadcast-view rally video in the sidebar to begin.")
    st.stop()

video_id = video_id_from_path(video_path)
corners_db = load_corners()
corners = corners_db.get(video_id)
summary = load_summary(video_id)

st.sidebar.markdown(f"**Video id:** `{video_id}`")
st.sidebar.markdown("**Court corners:** " + ("✅ set" if corners else "❌ not set"))
remark = st.sidebar.checkbox("Mark court corners", value=corners is None)
person_model = st.sidebar.selectbox("Player detector", ["yolov8n.pt (fast)", "yolov8x.pt (accurate, slow)"]).split()[0]
run = st.sidebar.button("▶ Run analysis", type="primary", disabled=corners is None and not remark)

st.title("Badminton Rally Analytics")

if remark:
    from streamlit_image_coordinates import streamlit_image_coordinates

    frame = first_frame(video_path)
    if frame is None:
        st.error("Could not read the video.")
        st.stop()
    key = f"clicks_{video_id}"
    st.session_state.setdefault(key, [])
    clicks = st.session_state[key]
    st.subheader("Mark the 4 outer court corners")
    nxt = CORNER_NAMES[len(clicks)] if len(clicks) < 4 else "done — press Save"
    st.write(f"Click order: far-left → far-right → near-right → near-left.  **Next: {nxt}**")

    display_w = 960
    scale = frame.shape[1] / display_w
    shown = frame.copy()
    for i, (x, y) in enumerate(clicks):
        cv2.circle(shown, (int(x), int(y)), 8, (255, 0, 0), -1)
        cv2.putText(shown, str(i + 1), (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if len(clicks) >= 2:
        cv2.polylines(shown, [np.array(clicks, dtype=np.int32)], len(clicks) == 4, (255, 255, 0), 2)
    value = streamlit_image_coordinates(shown, width=display_w, key=f"img_{video_id}_{len(clicks)}")
    if value and len(clicks) < 4:
        clicks.append((value["x"] * scale, value["y"] * scale))
        st.rerun()
    c1, c2 = st.columns(2)
    if c1.button("Undo last point") and clicks:
        clicks.pop()
        st.rerun()
    if c2.button("Save corners", disabled=len(clicks) != 4):
        corners_db[video_id] = [list(map(float, p)) for p in clicks]
        save_corners(corners_db)
        st.session_state[key] = []
        st.success("Corners saved. Untick 'Mark court corners' and press Run analysis.")
    st.stop()

if run:
    from analyze_match import analyze_video

    bar = st.progress(0.0, text="Starting…")
    with st.spinner("Analysing rally (players, shuttle, shots, predictions)…"):
        summary = analyze_video(video_path, corners=corners, person_model=person_model,
                                progress=lambda p, msg: bar.progress(min(float(p), 1.0), text=msg))
    bar.empty()

if summary is None:
    st.info("Press **▶ Run analysis** in the sidebar. The sample rally takes about a minute on CPU.")
    st.stop()

# ---------- results ----------
events = pd.DataFrame(summary["events"])
metrics = load_metrics()
info = summary["info"]

m = st.columns(6)
m[0].metric("Shots", summary["n_shots"])
m[1].metric("Rally length", f"{info['n_frames'] / info['fps']:.1f} s")
m[2].metric("Avg shot speed", f"{summary['avg_shot_speed_kmh']:.0f} km/h")
m[3].metric("Max shot speed", f"{summary['max_shot_speed_kmh']:.0f} km/h")
acc = summary.get("prediction_accuracy")
m[4].metric("Next-zone hits (this rally)", "—" if acc is None else f"{acc:.0%}")
if metrics:
    rf, mk = metrics["random_forest"], metrics["markov_baseline"]
    m[5].metric("Model accuracy (unseen rallies)", f"{rf['accuracy']:.0%}", f"{(rf['accuracy'] - mk['accuracy']) * 100:+.0f} pts vs baseline")

tab_video, tab_shots, tab_zones, tab_pred, tab_players = st.tabs(
    ["🎥 Annotated video", "📈 Shots", "🗺️ Court zones", "🔮 Next-zone prediction", "🏃 Players"])

with tab_video:
    video_file = ANALYSIS_DIR / video_id / "annotated.mp4"
    if video_file.is_file():
        st.video(str(video_file))
    st.caption("White grid = 12 court zones · orange zone = predicted next zone (shown ~0.4 s after each hit) · "
               "green flash = zone the shot actually went to.")

with tab_shots:
    if events.empty:
        st.warning("No shots detected in this video.")
    else:
        events["time_s"] = events["frame"] / info["fps"]
        events["Player"] = "Player " + events["player"].astype(str)
        fig = px.bar(events, x="shot_no", y="speed_kmh", color="Player", text="zone",
                     color_discrete_map={"Player 1": PLAYER_COLORS[1], "Player 2": PLAYER_COLORS[2]},
                     labels={"shot_no": "Shot", "speed_kmh": "Speed after hit (km/h)"},
                     title="Shot speed through the rally (label = zone the shot was hit from)")
        st.plotly_chart(fig, use_container_width=True)
        if "shot_group" not in events:  # analysed before shot groups were added
            events["shot_group"], events["shot_group_conf"] = None, None
        table = events[["shot_no", "time_s", "Player", "zone", "shot_group", "shot_group_conf", "speed_kmh",
                        "next_zone", "predicted_zone", "correct"]]
        st.dataframe(table.rename(columns={"shot_no": "Shot", "time_s": "Time (s)", "zone": "Hit from",
                                           "shot_group": "Shot", "shot_group_conf": "Shot confidence",
                                           "speed_kmh": "Speed (km/h)", "next_zone": "Went to",
                                           "predicted_zone": "Predicted", "correct": "Prediction correct"}),
                     use_container_width=True, hide_index=True,
                     column_config={"Shot confidence": st.column_config.NumberColumn(format="percent")})
        st.caption("Shot: pose-based shot group (serve, clear/lift, smash, drop/net, drive/push, defensive), "
                   "adapted to broadcast footage from hand-labelled hits; empty when no model beat guessing.")
        csv_path = ANALYSIS_DIR / video_id / "events.csv"
        if csv_path.is_file():
            st.download_button("Download events.csv", csv_path.read_bytes(), file_name=f"{video_id}_events.csv")

with tab_zones:
    if events.empty:
        st.warning("No shots detected.")
    else:
        c1, c2 = st.columns(2)
        hit_counts = events["zone"].value_counts().to_dict()
        c1.plotly_chart(court_figure(hit_counts, "Where shots were played from"), use_container_width=False)
        land = events["next_zone"].dropna()
        c2.plotly_chart(court_figure(land.value_counts().to_dict(), "Where shots were sent to", "Blues"),
                        use_container_width=False)
        st.caption("Zones 1–3 are next to the net, 4–6 near the baseline (left → right as seen on camera).")

with tab_pred:
    if events.empty:
        st.warning("No shots detected.")
    else:
        shot = st.slider("Shot", 1, int(events["shot_no"].max()), 1)
        ev = events[events["shot_no"] == shot].iloc[0]
        probs = ev["prediction"]
        outline = {}
        if isinstance(ev.get("next_zone"), str):
            outline[ev["next_zone"]] = dict(color="#00FF66", width=5)
        outline[ev["zone"]] = dict(color="#FFD23F", width=5)
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(court_figure({z: p * 100 for z, p in probs.items()},
                                     f"Shot {shot}: Player {ev['player']} hits from {ev['zone']} (yellow)<br>"
                                     f"<sup>chance per zone · actual = green</sup>", "Purples", "{:.0f}%",
                                     outline=outline),
                        use_container_width=False)
        bars = pd.DataFrame({"zone": list(probs), "probability": list(probs.values())}).sort_values("zone")
        c2.plotly_chart(px.bar(bars, x="zone", y="probability", title="Predicted next zone",
                               range_y=[0, 1]).update_traces(marker_color="#7B2CBF"), use_container_width=True)
        verdict = "—" if ev["correct"] is None or (isinstance(ev["correct"], float) and np.isnan(ev["correct"])) \
            else ("✅ correct" if ev["correct"] else "❌ wrong")
        c2.markdown(f"**Predicted:** {ev['predicted_zone']}  ·  **Actual:** {ev.get('next_zone') or 'unknown'}  ·  {verdict}")

        st.divider()
        st.subheader("How good is the predictor?")
        if metrics:
            rows = [{"Method": name.replace("_", " ").title(), "Top-1 accuracy": metrics[key]["accuracy"],
                     "Top-2 accuracy": metrics[key]["top2_accuracy"]}
                    for key, name in (("random_forest", "random forest (ours)"), ("markov_baseline", "most common next zone"),
                                      ("majority_baseline", "always same zone"), ("random_chance", "random guess"))]
            dfm = pd.DataFrame(rows).melt(id_vars="Method", var_name="Metric", value_name="Accuracy")
            st.plotly_chart(px.bar(dfm, x="Method", y="Accuracy", color="Metric", barmode="group", range_y=[0, 1],
                                   title=f"Accuracy on unseen rallies — {metrics['n_shots']} shots from "
                                         f"{metrics['n_videos']} rallies, 5-fold cross-validation by rally"),
                            use_container_width=True)
            imp = pd.DataFrame(list(metrics["feature_importance"].items()), columns=["feature", "importance"])
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.bar(imp.head(10), x="importance", y="feature", orientation="h",
                                   title="What the model looks at").update_yaxes(autorange="reversed"),
                            use_container_width=True)
            trans = metrics.get("training_transitions", {})
            if trans:
                mat = pd.DataFrame(0, index=ALL_ZONES, columns=ALL_ZONES)
                for key, count in trans.items():
                    a, b = key.split("->")
                    mat.loc[a, b] = count
                c2.plotly_chart(px.imshow(mat, text_auto=True, color_continuous_scale="Viridis",
                                          labels=dict(x="Next zone", y="Hit from", color="Shots"),
                                          title="Zone → next zone in the training rallies"),
                                use_container_width=True)
        else:
            st.info("No trained model yet — run `python train_zone_predictor.py`.")

with tab_players:
    players = summary["players"]
    c = st.columns(2)
    for col, pid in zip(c, ("1", "2")):
        p = players.get(pid) or players.get(int(pid), {})
        col.markdown(f"### Player {pid} ({'far' if pid == '1' else 'near'} side)")
        col.metric("Distance covered", f"{p.get('distance_m', 0):.1f} m")
        col.metric("Average movement speed", f"{p.get('avg_speed_mps', 0):.2f} m/s")
        col.plotly_chart(court_figure(p.get("zones", {}), f"Player {pid}: time spent per zone (frames)",
                                      "Reds" if pid == "1" else "Blues", height=440), use_container_width=False)
    fig = court_figure(title="Player movement on court", height=600)
    for pid, pts in summary["player_positions"].items():
        if pts:
            arr = np.array(pts)
            fig.add_trace(go.Scatter(x=arr[:, 1], y=arr[:, 2], mode="lines+markers", name=f"Player {pid}",
                                     marker=dict(size=4, color=PLAYER_COLORS[int(pid)]),
                                     line=dict(color=PLAYER_COLORS[int(pid)], width=2)))
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=False)
