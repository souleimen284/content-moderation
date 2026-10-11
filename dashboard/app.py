"""
Streamlit dashboard for the content moderation pipeline.

Connects directly to the same database the FastAPI backend writes to
(``settings.database_url``) via the shared analytics/queue services, so
every number here matches what ``GET /metrics`` reports — no separate
data path to keep in sync.

Run with:
    streamlit run dashboard/app.py

Sections (left sidebar):
  - Overview            : totals, status/content-type breakdown
  - Model Performance    : precision/recall/F1 per category + confusion matrix
  - Category Distribution: which policy categories fire most often
  - Volume & Confidence   : daily volume trend, confidence histograms
  - Review Queue          : interactive human-in-the-loop review
  - Moderator Performance : review counts, decision mix, avg latency
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `streamlit run dashboard/app.py` from the project root
# without needing to install the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.core.config import settings
from app.db.database import session_scope
from sqlalchemy import select

from app.db.models import Modality, ReviewDecision, User, UserRole
from app.services.analytics_service import (
    get_active_model_confusion_matrices,
    get_all_model_versions,
    get_category_distribution,
    get_confidence_scores,
    get_daily_volume,
    get_moderator_performance,
    get_overall_metrics,
)
from app.services.queue_service import list_queue, submit_review

st.set_page_config(page_title="Content Moderation Dashboard", layout="wide", page_icon="🛡️")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    /* ── Night-mode palette ── */
    :root {
        --bg:       #0e1117;
        --surface:  #1a1d24;
        --card:     #21262d;
        --border:   #30363d;
        --text:     #e6edf3;
        --muted:    #8b949e;
        --mint:     #3fb950;
        --coral:    #f85149;
        --gold:     #d29922;
        --accent:   #58a6ff;
    }

    /* ── Base typography ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--text) !important;
    }
    h1, h2, h3, h4 {
        color: var(--text) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -.025em;
    }

    /* ── Main container ── */
    [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
    }
    .block-container {
        max-width: 1440px;
        padding: 2.2rem 3rem 3rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] .stRadio label {
        padding: 7px 10px;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,.08);
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: var(--card) !important;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,.3);
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .78rem; }
    [data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stMetricLabel"] * { color: var(--muted) !important; }

    /* ── Text / Markdown ── */
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stText, label,
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {
        color: var(--text) !important;
    }

    /* ── Inputs / selects ── */
    input, textarea,
    [data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        color: var(--text) !important;
        background: var(--surface) !important;
        border-color: var(--border) !important;
    }
    [data-baseweb="select"] * { color: var(--text) !important; }
    [data-baseweb="option"] { color: var(--text) !important; background: var(--surface) !important; }
    [data-baseweb="menu"] * { color: var(--text) !important; background: var(--surface) !important; }
    [data-baseweb="tag"] { color: var(--text) !important; background: var(--card) !important; }

    /* ── DataFrame / table ── */
    [data-testid="stDataFrame"] { background: var(--card) !important; }
    [data-testid="stDataFrame"] * { color: var(--text) !important; }
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
        color: var(--text) !important;
        background: var(--card) !important;
    }
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrame"] [role="columnheader"] {
        color: var(--text) !important;
    }
    [data-baseweb="table"] * { color: var(--text) !important; }

    /* ── Alerts / captions ── */
    [data-testid="stAlert"] p, [data-testid="stCaptionContainer"] p,
    [data-testid="stCaption"] { color: var(--muted) !important; }
    .stInfo p, .stSuccess p, .stWarning p, .stError p { color: var(--text) !important; }

    /* ── Buttons ── */
    button[kind="secondary"] {
        color: var(--text) !important;
        background: var(--card) !important;
        border-color: var(--border) !important;
    }
    button[kind="primary"] {
        color: #ffffff !important;
        background: var(--mint) !important;
    }

    /* ── Widget labels ── */
    .stRadio label, .stRadio label p,
    .stSelectbox label, .stSelectbox label p,
    .stMultiSelect label, .stMultiSelect label p,
    .stSlider label, .stSlider label p,
    .stCheckbox label, .stCheckbox label p {
        color: var(--text) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] { color: var(--muted) !important; }
    .stTabs [aria-selected="true"] { color: var(--text) !important; }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: var(--card) !important;
        border: 1px solid var(--border);
        border-radius: 14px;
    }
    [data-testid="stExpander"] * { color: var(--text) !important; }
    [data-testid="stExpander"] summary p { color: var(--text) !important; }
    [data-testid="stHorizontalBlock"] * { color: var(--text) !important; }

    /* ── Custom cards / panels ── */
    .page-kicker {
        color: var(--mint);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .25rem;
    }
    .page-subtitle { color: var(--muted); margin-top: -.6rem; margin-bottom: 1.6rem; }
    .signal-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 15px 17px;
        margin: .35rem 0;
    }
    .signal-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: .88rem;
        font-weight: 600;
        color: var(--text);
    }
    .signal-head span, .signal-head span * { color: var(--text) !important; }
    .signal-meta { color: var(--muted); font-size: .76rem; }
    .signal-track {
        height: 7px;
        background: var(--border);
        border-radius: 10px;
        overflow: hidden;
        margin-top: 9px;
    }
    .signal-fill { height: 100%; border-radius: 10px; background: var(--mint); }
    .signal-fill.alert { background: var(--coral); }
    .queue-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 4px 10px;
        margin: .7rem 0;
    }
    .queue-card * { color: var(--text) !important; }
    .model-badge {
        display: inline-block;
        color: var(--mint);
        background: rgba(63,185,80,.12);
        border-radius: 999px;
        padding: 4px 9px;
        font-size: .72rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )


def page_intro(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="page-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def chart(fig: go.Figure, height: int = 330) -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#8b949e"),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def signal_panel(scores: dict[str, float], flagged: list[str] | None = None) -> None:
    flagged = flagged or []
    if not scores:
        st.caption("No category-level image signals were recorded.")
        return
    for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        safe_label = label.replace("_", " ").title()
        alert = label in flagged or score >= settings.image_nsfw_threshold
        st.markdown(
            f'<div class="signal-card"><div class="signal-head"><span>{safe_label}</span>'
            f'<span class="signal-meta">{score:.1%}</span></div>'
            f'<div class="signal-track"><div class="signal-fill {"alert" if alert else ""}" style="width:{score * 100:.1f}%"></div></div></div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("🛡️ Moderation Dashboard")
st.sidebar.caption(f"Connected to: `{settings.database_url.split('@')[-1] if '@' in settings.database_url else settings.database_url}`")

section = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Model Performance",
        "Category Distribution",
        "Volume & Confidence",
        "Review Queue",
        "Moderator Performance",
    ],
)

if st.sidebar.button("🔄 Refresh data"):
    st.rerun()


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def render_overview() -> None:
    page_intro("Command center", "Moderation overview", "A live read on traffic, review pressure, and automated decisions.")

    with session_scope() as db:
        stats = get_overall_metrics(db)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total requests", f"{stats['total_requests']:,}")
    col2.metric("Needs review", f"{stats['queue_size']:,}")
    col3.metric("Reviewed", f"{stats['reviewed_count']:,}")
    auto_resolved = stats["requests_by_status"].get("auto_approved", 0) + stats["requests_by_status"].get(
        "auto_rejected", 0
    )
    automation_rate = (auto_resolved / stats["total_requests"] * 100) if stats["total_requests"] else 0
    col4.metric("Automation rate", f"{automation_rate:.1f}%", help="Auto-approved and auto-rejected requests.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Requests by status")
        status_df = pd.DataFrame(
            list(stats["requests_by_status"].items()), columns=["status", "count"]
        )
        if not status_df.empty:
            fig = px.pie(status_df, names="status", values="count", hole=0.58, color_discrete_sequence=["#20a77a", "#e05a68", "#d2942c", "#71809b"])
            chart(fig, 310)
        else:
            st.info("No moderation requests yet.")

    with c2:
        st.subheader("Requests by content type")
        ct_df = pd.DataFrame(
            list(stats["requests_by_content_type"].items()), columns=["content_type", "count"]
        )
        if not ct_df.empty:
            fig = px.bar(ct_df, x="content_type", y="count", color="content_type", color_discrete_sequence=["#2b8f75", "#e09a3e", "#71809b"])
            chart(fig, 310)
        else:
            st.info("No moderation requests yet.")

    st.subheader("Moderator decisions")
    if stats["reviews_by_decision"]:
        dec_df = pd.DataFrame(list(stats["reviews_by_decision"].items()), columns=["decision", "count"])
        chart(px.bar(dec_df, x="decision", y="count", color="decision", color_discrete_sequence=["#20a77a", "#e05a68", "#d2942c"]), 290)
    else:
        st.info("No moderator reviews submitted yet.")


# --------------------------------------------------------------------------- #
# Model performance: precision/recall/F1 + confusion matrix per category
# --------------------------------------------------------------------------- #
def _render_model_metrics(model_label: str, modality: Modality) -> None:
    st.subheader(f"{model_label} model")
    with session_scope() as db:
        versions = get_all_model_versions(db, modality)
        # Detach the fields we need into plain dicts before the session closes.
        version_rows = [
            {
                "version_tag": v.version_tag,
                "is_active": v.is_active,
                "trained_at": v.trained_at,
                "training_samples": v.training_samples,
                "metrics": v.metrics or {},
            }
            for v in versions
        ]

    if not version_rows:
        st.info(f"No {model_label.lower()} model has been trained/registered yet.")
        return

    active = next((v for v in version_rows if v["is_active"]), version_rows[0])
    st.caption(
        f"Active version: **{active['version_tag']}** · trained {active['trained_at']:%Y-%m-%d %H:%M UTC} "
        f"· {active['training_samples']} training samples"
    )

    metrics = active["metrics"]
    labels = settings.text_labels if modality == Modality.TEXT else settings.image_labels

    agg_cols = st.columns(3)
    agg_cols[0].metric("F1 (macro)", f"{metrics.get('f1_macro', 0):.3f}")
    agg_cols[1].metric("Precision (macro)", f"{metrics.get('precision_macro', 0):.3f}")
    agg_cols[2].metric("Recall (macro)", f"{metrics.get('recall_macro', 0):.3f}")

    per_label_rows = []
    for label in labels:
        per_label_rows.append(
            {
                "category": label,
                "precision": metrics.get(f"precision_{label}"),
                "recall": metrics.get(f"recall_{label}"),
                "f1": metrics.get(f"f1_{label}"),
                "support": metrics.get(f"support_{label}"),
            }
        )
    per_label_df = pd.DataFrame(per_label_rows)
    if per_label_df["precision"].notna().any():
        st.dataframe(per_label_df, hide_index=True)

        melted = per_label_df.melt(
            id_vars="category", value_vars=["precision", "recall", "f1"], var_name="metric", value_name="score"
        )
        st.plotly_chart(
            px.bar(melted, x="category", y="score", color="metric", barmode="group"),
        )
    else:
        st.info("No per-category metrics recorded for this model version yet.")

    confusion = metrics.get("confusion_matrix")
    if confusion:
        st.markdown("**Confusion matrices (per category)**")
        cm_cols = st.columns(min(3, len(labels)))
        for i, label in enumerate(labels):
            cm = confusion.get(label)
            if not cm:
                continue
            z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
            fig = go.Figure(
                data=go.Heatmap(
                    z=z,
                    x=["Pred: No", "Pred: Yes"],
                    y=["Actual: No", "Actual: Yes"],
                    text=z,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=False,
                )
            )
            fig.update_layout(title=label, height=250, margin=dict(l=10, r=10, t=40, b=10))
            with cm_cols[i % len(cm_cols)]:
                st.plotly_chart(fig)

    if len(version_rows) > 1:
        with st.expander(f"Version history ({len(version_rows)} versions)"):
            hist_df = pd.DataFrame(
                [
                    {
                        "version": v["version_tag"],
                        "active": v["is_active"],
                        "trained_at": v["trained_at"],
                        "f1_macro": v["metrics"].get("f1_macro"),
                        "training_samples": v["training_samples"],
                    }
                    for v in version_rows
                ]
            )
            st.dataframe(hist_df, hide_index=True)


def render_model_performance() -> None:
    page_intro("Quality lab", "Model performance", "Training metrics for text and image models, with the active checkpoint called out clearly.")
    _render_model_metrics("Text moderation (BERT)", Modality.TEXT)
    st.divider()
    _render_model_metrics("Image moderation (NudeNet + CLIP signals)", Modality.IMAGE)


# --------------------------------------------------------------------------- #
# Category distribution
# --------------------------------------------------------------------------- #
def render_category_distribution() -> None:
    page_intro("Policy lens", "Category distribution", "See which policy signals are appearing most often in live moderation traffic.")

    c1, c2 = st.columns(2)
    with session_scope() as db:
        text_dist = get_category_distribution(db, Modality.TEXT)
        image_dist = get_category_distribution(db, Modality.IMAGE)

    with c1:
        st.subheader("Text categories")
        if not text_dist.empty:
            # Filter out stale labels not in the current config.
            text_dist = text_dist[text_dist["label"].isin(settings.text_labels)]
        if not text_dist.empty:
            chart(px.bar(text_dist, x="label", y="count", color="label", color_discrete_sequence=["#5573b8"]), 330)
        else:
            st.info("No flagged text predictions yet.")

    with c2:
        st.subheader("Image categories")
        if not image_dist.empty:
            # Filter out stale labels not in the current config.
            image_dist = image_dist[image_dist["label"].isin(settings.image_labels)]
            image_dist = image_dist.copy()
            image_dist["label"] = image_dist["label"].str.replace("_", " ").str.title()
        if not image_dist.empty:
            chart(px.bar(image_dist, x="label", y="count", color="label", color_discrete_sequence=["#e05a68", "#d2942c", "#2b8f75", "#5573b8"]), 330)
        else:
            st.info("No flagged image predictions yet.")


# --------------------------------------------------------------------------- #
# Volume & confidence
# --------------------------------------------------------------------------- #
def render_volume_and_confidence() -> None:
    page_intro("Operations", "Volume & confidence", "Track throughput and understand where model scores cluster around review thresholds.")

    days = st.slider("Days of history", min_value=7, max_value=90, value=30)
    with session_scope() as db:
        volume_df = get_daily_volume(db, days=days)
        text_scores = get_confidence_scores(db, Modality.TEXT)
        image_scores = get_confidence_scores(db, Modality.IMAGE)

    st.subheader("Daily moderation volume")
    if not volume_df.empty:
        chart(px.bar(volume_df, x="date", y="count", color_discrete_sequence=["#2b8f75"]), 330)
    else:
        st.info("No requests in this time range.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Text model confidence distribution")
        if not text_scores.empty:
            chart(px.histogram(text_scores, x="top_score", nbins=20, range_x=[0, 1], color_discrete_sequence=["#5573b8"]), 280)
            _threshold_lines()
        else:
            st.info("No text predictions yet.")

    with c2:
        st.subheader("Image model confidence distribution")
        if not image_scores.empty:
            chart(px.histogram(image_scores, x="top_score", nbins=20, range_x=[0, 1], color_discrete_sequence=["#e05a68"]), 280)
            _threshold_lines()
        else:
            st.info("No image predictions yet.")


def _threshold_lines() -> None:
    st.caption(
        f"Routing thresholds: auto > {settings.auto_decision_threshold:.0%}, "
        f"queue {settings.queue_threshold:.0%}–{settings.auto_decision_threshold:.0%}, "
        f"uncertain < {settings.queue_threshold:.0%}"
    )


# --------------------------------------------------------------------------- #
# Review queue (interactive)
# --------------------------------------------------------------------------- #
def render_review_queue() -> None:
    page_intro("Human review", "Review queue", "Resolve uncertain cases with the model evidence visible beside the original content.")

    with session_scope() as db:
        items = list_queue(db, limit=100)
        # Detach the fields we need before the session closes.
        rows = [
            {
                "id": item.id,
                "content_type": item.content_type.value,
                "text_content": item.text_content,
                "image_path": item.image_path,
                "status": item.status.value,
                "created_at": item.created_at,
                "predictions": [
                    {
                        "modality": p.modality.value,
                        "top_label": p.top_label,
                        "top_score": p.top_score,
                        "predicted_labels": p.predicted_labels,
                        "model_version": p.model_version,
                        "label_scores": p.label_scores or {},
                    }
                    for p in item.predictions
                ],
            }
            for item in items
        ]

    st.caption(f"{len(rows)} item(s) awaiting review")
    if not rows:
        st.success("Queue is empty — nothing awaiting review right now.")
        return

    for row in rows:
        with st.expander(
            f"[{row['status'].upper()}] {row['content_type']} — {row['id'][:8]}… "
            f"({row['created_at']:%Y-%m-%d %H:%M UTC})"
        ):
            st.markdown('<div class="queue-card">', unsafe_allow_html=True)
            col_content, col_action = st.columns([2, 1])

            with col_content:
                if row["text_content"]:
                    st.markdown("**Text:**")
                    st.write(row["text_content"])
                if row["image_path"]:
                    st.markdown("**Image:**")
                    image_path = Path(row["image_path"])
                    if image_path.exists():
                        st.image(str(image_path), width=300)
                    else:
                        st.caption(f"(image file not found at {image_path})")

                st.markdown("**Model prediction(s):**")
                for pred in row["predictions"]:
                    # Filter out labels not in the current config (e.g. old
                    # categories like 'misinformation' that have been removed).
                    current_labels = (
                        settings.text_labels
                        if pred["modality"] == "text"
                        else settings.image_labels
                    )
                    valid_predicted = [
                        l for l in pred["predicted_labels"] if l in current_labels
                    ]
                    valid_scores = {
                        k: v for k, v in pred["label_scores"].items() if k in current_labels
                    }
                    top_label = pred["top_label"]
                    if top_label not in current_labels:
                        # Fall back to the highest-scored valid label
                        if valid_scores:
                            top_label = max(valid_scores, key=valid_scores.get)
                        else:
                            top_label = "(no valid labels)"
                    st.markdown(
                        f"**{pred['modality'].title()} signal** · `{pred['model_version']}`  \n"
                        f"Top signal: **{top_label.replace('_', ' ').title()}** · {pred['top_score']:.1%} · "
                        f"Flagged: {', '.join(valid_predicted) if valid_predicted else 'none'}"
                    )
                    if pred["modality"] == "image":
                        st.caption("Image signals")
                        signal_panel(valid_scores, valid_predicted)

            with col_action:
                st.markdown("**Moderator decision**")
                decision_label = st.radio(
                    "Decision", ["approve", "reject", "escalate"], key=f"decision_{row['id']}", horizontal=True
                )
                available_labels = (
                    settings.text_labels if row["content_type"] != "image" else settings.image_labels
                )
                corrected = st.multiselect(
                    "Corrected labels (ground truth)", options=available_labels, key=f"labels_{row['id']}"
                )
                notes = st.text_area("Notes", key=f"notes_{row['id']}", height=80)
                moderator_name = st.text_input(
                    "Moderator name/ID", key=f"mod_{row['id']}", value="dashboard_user"
                )

                if st.button("Submit review", key=f"submit_{row['id']}"):
                    with session_scope() as db:
                        try:
                            # Resolve moderator name to a User UUID.
                            # The moderator_reviews table has a FK to users,
                            # so we look up or auto-create the user first.
                            resolved_moderator_id = None
                            if moderator_name:
                                stmt = select(User).where(User.username == moderator_name)
                                user = db.scalars(stmt).first()
                                if user is None:
                                    user = User(
                                        username=moderator_name,
                                        email=f"{moderator_name}@dashboard.local",
                                        hashed_password="dashboard-internal",
                                        role=UserRole.MODERATOR,
                                    )
                                    db.add(user)
                                    db.flush()
                                resolved_moderator_id = user.id

                            submit_review(
                                db,
                                moderation_request_id=row["id"],
                                decision=ReviewDecision(decision_label),
                                moderator_id=resolved_moderator_id,
                                notes=notes or None,
                                corrected_labels=corrected or None,
                            )
                            st.success("Review submitted.")
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
            st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Moderator performance
# --------------------------------------------------------------------------- #
def render_moderator_performance() -> None:
    page_intro("Team health", "Moderator performance", "Review throughput, decisions, and average time from intake to resolution.")

    with session_scope() as db:
        perf_df = get_moderator_performance(db)

    if perf_df.empty:
        st.info("No moderator reviews submitted yet.")
        return

    display_df = perf_df.copy()
    display_df["avg_review_seconds"] = display_df["avg_review_seconds"].round(1)
    st.dataframe(display_df, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Reviews per moderator")
        chart(px.bar(perf_df, x="moderator_id", y="total_reviews", color_discrete_sequence=["#2b8f75"]), 300)
    with c2:
        st.subheader("Average review time (seconds)")
        chart(px.bar(perf_df, x="moderator_id", y="avg_review_seconds", color_discrete_sequence=["#d2942c"]), 300)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
_SECTIONS = {
    "Overview": render_overview,
    "Model Performance": render_model_performance,
    "Category Distribution": render_category_distribution,
    "Volume & Confidence": render_volume_and_confidence,
    "Review Queue": render_review_queue,
    "Moderator Performance": render_moderator_performance,
}

_SECTIONS[section]()
