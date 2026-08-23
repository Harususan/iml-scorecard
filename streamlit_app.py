#!/usr/bin/env python3
"""
streamlit_app.py — FPL H2H Scout (Streamlit UI)
================================================
A Streamlit front-end for fpl_h2h_v14.py. All FPL API calls, scoring math,
captain-multiplier logic, and PDF rendering live in that module unchanged —
this file only handles input widgets, caching, and displaying the results.

Run with:
    streamlit run streamlit_app.py

Requires fpl_h2h_v14.py to sit in the same folder (it's imported as a module,
not executed — its own `if __name__ == "__main__"` guard keeps the CLI path
from firing here).
"""

import contextlib
import io
import json
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fpl_h2h_v14 as engine

st.set_page_config(page_title="FPL H2H Scout", page_icon="⚽", layout="wide")

CHIP_STYLE = {
    "available": "background-color: #e6f7f5; color: #0f766e;",
    "used":      "background-color: #eef0f4; color: #64748b; text-decoration: line-through;",
    "active":    "background-color: #e0e7ff; color: #4338ca; font-weight: 700;",
}

# ── League registry ────────────────────────────────────────────────────────────
# Each "team" here is a 4-manager classic mini-league named after its PL club.
# Selecting a team in the sidebar resolves straight to its League ID and
# manager roster — no need to type/paste IDs or guess a captain's name.
TEAMS = {
    "Arsenal":                 {"league_id": "767705",  "managers": ["Aayush Bermola", "Bilal Mahmood", "Saksham Mishra", "Tanish Bermola"]},
    "Aston Villa":              {"league_id": "705859",  "managers": ["Agneebha Ghosh", "Subhajit Dutta", "Ishaan Goel", "Vinamra Dave"]},
    "Bournemouth":              {"league_id": "729033",  "managers": ["Sarthak Grover", "Soumil Mendiratta", "Parameshwar Hembram", "Chandrashekhar Ramadoss"]},
    "Brentford":                {"league_id": "705820",  "managers": ["Sannan Shah", "Sheikh Usmaan", "Amaan Seven", "Aflaq Shah"]},
    "Brighton and Hove Albion": {"league_id": "906398",  "managers": ["Mukul Kundu", "Nishanth G Suseelan", "Shivam Pahuja", "Abhijeet Kundu"]},
    "Chelsea":                  {"league_id": "1560363", "managers": ["Shashwat Prakash Dubey", "Amitash Srivastava", "Sourav Hemran", "Winayak Kumar"]},
    "Coventry City":            {"league_id": "1388324", "managers": ["Sreejan Deb", "Sunny Das", "Debanjan Dutta", "Rohit Sengupta"]},
    "Crystal Palace":           {"league_id": "944252",  "managers": ["Arish Mehta", "Aryaman Arora", "sidhanth muralidhar", "Farhan ul Haq"]},
    "Everton":                  {"league_id": "706707",  "managers": ["Ayush Falor", "Dylan M", "Gau Mohanty", "Rizwan Azavedo"]},
    "Fulham":                   {"league_id": "1169975", "managers": ["Sehaj Singh", "Piravinthan Susendralingam", "Jawad Ali", "Samarth Vishal Sood"]},
    "Hull City":                {"league_id": "708601",  "managers": ["Vikhayat Arora", "Mehul Goyal", "Aayush Sahanan", "Animesh Vijayvargiya"]},
    "Ipswich Town":             {"league_id": "1564554", "managers": ["Darshan Lakhani", "Bishpan Singh", "Badal Kumar", "Tushar Gupta"]},
    "Leeds United":             {"league_id": "1546694", "managers": ["Rohan Biswas", "Bhagat Khatiwoda", "Aman Gupta", "Sayantan Mondal"]},
    "Liverpool":                {"league_id": "1658463", "managers": ["Rudrabha Chakraborty", "Unmesh Gavand", "Mohan Arora", "Shashank n"]},
    "Manchester City":          {"league_id": "972886",  "managers": ["Harsh Bhat", "Priyanshu Mukherjee", "Uday Chandak", "Nimish Sanghavi"]},
    "Manchester United":        {"league_id": "1260458", "managers": ["Advaita Gupta", "Vasav Gupta", "Arudra Sen Gupta", "Raghav Datta"]},
    "Newcastle United":         {"league_id": "1124106", "managers": ["Hussain Jawadwala", "Rithvik R", "Krutik Patel", "Dk Vudatha"]},
    "Nottingham Forest":        {"league_id": "710305",  "managers": ["Anmol Gupta", "Avi Naik", "Abhishek Pedamkar", "Dibyendu Adhikary"]},
    "Sunderland":               {"league_id": "1498809", "managers": ["Vaibhav Garg", "Naman Bhatia", "Gaurav Joshi", "Jai Gupta"]},
    "Tottenham Hotspurs":       {"league_id": "1547761", "managers": ["Hitesh Kumar", "Ketan Virmani", "Akshay Soni", "Hriday Ranade"]},
}
TEAM_NAMES = sorted(TEAMS.keys())


# ── Cached wrappers around the engine's network calls ────────────────────────
# (kept separate from engine.py itself so the CLI script stays untouched)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_bootstrap():
    return engine.get_bootstrap()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_player_map(_bootstrap):
    return engine.build_player_map(_bootstrap)


@st.cache_data(ttl=60, show_spinner=False)
def cached_live_scores(gw: int):
    return engine.get_live_scores(gw)


@st.cache_data(ttl=300, show_spinner=False)
def cached_league_name(league_id: str):
    return engine.get_league_name(league_id)


@st.cache_data(ttl=300, show_spinner=False)
def cached_league_managers(league_id: str):
    return engine.get_league_managers(league_id)


@st.cache_data(ttl=120, show_spinner=False)
def cached_team_picks(manager_ids, cap_index, gw, team_label, _player_map, managers):
    return engine.fetch_team_picks(manager_ids, cap_index, gw, team_label, _player_map, managers)


@st.cache_data(ttl=120, show_spinner=False)
def cached_fixture_kickoffs(gw: int):
    return engine.get_fixture_kickoffs(gw)


@st.cache_data(ttl=60, show_spinner=False)
def cached_fixture_status(gw: int):
    """club_id -> 'finished' | 'live' | 'upcoming', from the raw fixtures endpoint."""
    try:
        data = engine.fetch(f"{engine.BASE}/fixtures/?event={gw}")
    except Exception:
        return {}
    status = {}
    for f in data:
        if f.get("finished"):
            s = "finished"
        elif f.get("started"):
            s = "live"
        else:
            s = "upcoming"
        for tid in (f.get("team_h"), f.get("team_a")):
            if tid:
                status[tid] = s
    return status


@st.cache_data(ttl=300, show_spinner=False)
def cached_histories(manager_ids):
    return engine.get_all_season_histories(manager_ids)


# ── Pipeline: mirrors main() in fpl_h2h_v14.py, minus argparse/printing ──────

def run_pipeline(gw, league_a, league_b, cap_a_query, cap_b_query, no_live):
    """Runs the full fetch + compute pipeline. Raises SystemExit (caught by
    the caller) on bad captain names, same as the CLI does."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bootstrap  = cached_bootstrap()
        player_map = cached_player_map(bootstrap)

        if no_live:
            live_scores, live_explain = {}, {}
        else:
            live_scores, _, live_explain = cached_live_scores(gw)

        league_name_a = cached_league_name(league_a)
        managers_a    = cached_league_managers(league_a)
        league_name_b = cached_league_name(league_b)
        managers_b    = cached_league_managers(league_b)

        cap_a_idx = engine.resolve_cap_index(cap_a_query, managers_a, "A")
        cap_b_idx = engine.resolve_cap_index(cap_b_query, managers_b, "B")

        team_a_ids = [m["id"] for m in managers_a]
        team_b_ids = [m["id"] for m in managers_b]

        picks_a = cached_team_picks(team_a_ids, cap_a_idx, gw, "A", player_map, managers_a)
        picks_b = cached_team_picks(team_b_ids, cap_b_idx, gw, "B", player_map, managers_b)

        histories_a  = cached_histories(team_a_ids)
        histories_b  = cached_histories(team_b_ids)
        chips_hist_a = {mid: h["chips"] for mid, h in histories_a.items()}
        chips_hist_b = {mid: h["chips"] for mid, h in histories_b.items()}

        rows = engine.build_differential(picks_a, picks_b, player_map, live_scores)

    return dict(
        gw=gw, player_map=player_map, live_scores=live_scores, live_explain=live_explain,
        league_name_a=league_name_a, league_name_b=league_name_b,
        managers_a=managers_a, managers_b=managers_b,
        cap_a_idx=cap_a_idx, cap_b_idx=cap_b_idx,
        team_a_ids=team_a_ids, team_b_ids=team_b_ids,
        picks_a=picks_a, picks_b=picks_b,
        histories_a=histories_a, histories_b=histories_b,
        chips_hist_a=chips_hist_a, chips_hist_b=chips_hist_b,
        rows=rows,
    )


def team_total(picks_list, cap_idx, live_scores):
    total = 0
    for i, mgr in enumerate(picks_list):
        _, _, final = engine.manager_final_score(mgr, live_scores)
        total += final * 2 if i == cap_idx else final
    return total


# ── Sidebar: match settings ───────────────────────────────────────────────────

with st.sidebar:
    st.header("⚽ Match settings")
    gw = st.number_input("Gameweek", min_value=1, max_value=38, value=1, step=1)

    team_a_name = st.selectbox("Team A", TEAM_NAMES, index=0)
    cap_a = st.selectbox(f"{team_a_name} H2H captain", TEAMS[team_a_name]["managers"])

    st.write("")
    b_default = 1 if TEAM_NAMES[0] == team_a_name else 0
    team_b_name = st.selectbox("Team B", TEAM_NAMES, index=b_default)
    cap_b = st.selectbox(f"{team_b_name} H2H captain", TEAMS[team_b_name]["managers"])

    league_a = TEAMS[team_a_name]["league_id"]
    league_b = TEAMS[team_b_name]["league_id"]

    st.divider()
    no_live    = st.checkbox("Skip live scores", value=False,
                              help="Use for a season that's fully finished, or to speed up testing.")
    no_summary = st.checkbox("Skip squad summaries", value=False)
    no_chips   = st.checkbox("Skip chip tracker", value=False)

    st.divider()
    run_clicked = st.button("Run analysis", type="primary", width="stretch")

if "results" not in st.session_state:
    st.session_state.results = None
if "settings" not in st.session_state:
    st.session_state.settings = None

if run_clicked:
    if team_a_name == team_b_name:
        st.error("Team A and Team B must be different clubs.")
    else:
        try:
            with st.spinner("Talking to the FPL API — leagues, picks, chip history..."):
                st.session_state.results = run_pipeline(
                    int(gw), league_a, league_b, cap_a, cap_b, no_live,
                )
                st.session_state.settings = dict(no_summary=no_summary, no_chips=no_chips)
            st.toast("Analysis complete!", icon="✅")
        except SystemExit as e:
            st.error(str(e).strip() or "Couldn't resolve one of the captains — check the roster and try again.")
        except Exception as e:
            st.error(f"Something went wrong fetching data: {e}")

st.title("FPL H2H Scout")
st.caption("Head-to-head ownership & point-swing analyser — 2026/27 season")

res = st.session_state.results
if res is None:
    st.info("Pick both teams and their H2H captains in the sidebar, then click **Run analysis**.")
    st.stop()

settings      = st.session_state.settings
gw            = res["gw"]
player_map    = res["player_map"]
live_scores   = res["live_scores"]
league_name_a = res["league_name_a"]
league_name_b = res["league_name_b"]
managers_a, managers_b = res["managers_a"], res["managers_b"]
cap_a_idx, cap_b_idx   = res["cap_a_idx"], res["cap_b_idx"]
picks_a, picks_b       = res["picks_a"], res["picks_b"]
chips_hist_a, chips_hist_b = res["chips_hist_a"], res["chips_hist_b"]
histories_a, histories_b   = res["histories_a"], res["histories_b"]
rows          = res["rows"]

total_a = team_total(picks_a, cap_a_idx, live_scores)
total_b = team_total(picks_b, cap_b_idx, live_scores)

tab_names = ["Scoreboard", "GW Race", "Team Totals", "Differential & Swing", "Season Trend"]
if not settings["no_chips"]:
    tab_names.append("Chip Tracker")
if not settings["no_summary"]:
    tab_names.append("Squads")
tabs = dict(zip(tab_names, st.tabs(tab_names)))


# ── Scoreboard ────────────────────────────────────────────────────────────────
with tabs["Scoreboard"]:
    st.subheader(f"{league_name_a}  vs  {league_name_b}  —  GW{gw}")
    c1, c2, c3 = st.columns([2, 1, 2])
    c1.metric(league_name_a, total_a)
    diff = total_a - total_b
    if diff > 0:
        c2.metric("Lead", f"+{diff}", delta=f"{league_name_a}")
    elif diff < 0:
        c2.metric("Lead", f"+{abs(diff)}", delta=f"{league_name_b}")
    else:
        c2.metric("Lead", "Level")
    c3.metric(league_name_b, total_b)

    total = total_a + total_b
    pct_a = (total_a / total * 100) if total else 50
    fig = go.Figure()
    fig.add_bar(x=[pct_a], y=["Share"], orientation="h", name=league_name_a,
                marker_color="#1a8754", text=[f"{pct_a:.0f}%"], textposition="inside")
    fig.add_bar(x=[100 - pct_a], y=["Share"], orientation="h", name=league_name_b,
                marker_color="#0e7490", text=[f"{100-pct_a:.0f}%"], textposition="inside")
    fig.update_layout(barmode="stack", height=90, showlegend=True,
                       margin=dict(l=0, r=0, t=10, b=10), xaxis=dict(visible=False),
                       yaxis=dict(visible=False))
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.markdown("##### Downloads")
    dl_col1, dl_col2 = st.columns(2)

    pdf_args = SimpleNamespace(
        gw=gw, team_a=res["team_a_ids"], team_b=res["team_b_ids"],
        league_name_a=league_name_a, league_name_b=league_name_b,
        no_summary=settings["no_summary"],
    )
    pdf_buf = io.BytesIO()
    try:
        engine.generate_pdf_report(
            pdf_buf, pdf_args, picks_a, picks_b, player_map, live_scores, rows,
            cap_a_idx, cap_b_idx, managers_a=managers_a, managers_b=managers_b,
            chips_hist_a=None if settings["no_chips"] else chips_hist_a,
            chips_hist_b=None if settings["no_chips"] else chips_hist_b,
        )
        dl_col1.download_button(
            "Download PDF report", data=pdf_buf.getvalue(),
            file_name=f"fpl_h2h_gw{gw}.pdf", mime="application/pdf",
            width="stretch",
        )
    except SystemExit as e:
        dl_col1.warning(str(e))

    json_payload = json.dumps({
        "gw": gw, "league_a": league_name_a, "league_b": league_name_b,
        "total_a": total_a, "total_b": total_b, "differential": rows,
    }, indent=2)
    dl_col2.download_button(
        "Download JSON data", data=json_payload,
        file_name=f"fpl_h2h_gw{gw}.json", mime="application/json",
        width="stretch",
    )


# ── Team totals ───────────────────────────────────────────────────────────────
with tabs["Team Totals"]:
    for label, picks_list, cap_idx, managers, league_nm, grand_total in (
        ("A", picks_a, cap_a_idx, managers_a, league_name_a, total_a),
        ("B", picks_b, cap_b_idx, managers_b, league_name_b, total_b),
    ):
        st.markdown(f"##### {league_nm} (Team {label})")
        recs = []
        for i, mgr in enumerate(picks_list):
            raw, hit, final = engine.manager_final_score(mgr, live_scores)
            is_cap = (i == cap_idx)
            contribution = final * 2 if is_cap else final
            mgr_label = engine._mgr_label(managers, i, mgr["manager_id"])
            recs.append({
                "Manager": mgr_label, "Raw": raw, "Hit": hit, "Final": final,
                "H2H Cap": "★ x2" if is_cap else "—", "Contribution": contribution,
            })
        df = pd.DataFrame(recs)
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption(f"Team {label} total: **{grand_total}** pts")
        st.write("")


# ── Differential & swing ──────────────────────────────────────────────────────
STATUS_LABEL = {"finished": "✅ FT", "live": "🔴 LIVE", "upcoming": "⏳ Upcoming", "unknown": "❔ —"}
STATUS_ORDER = ["live", "upcoming", "finished", "unknown"]
NAME_TO_TEAM_ID = {v: k for k, v in engine.CLUBS.items()}

with tabs["Differential & Swing"]:
    st.markdown(f"##### Ownership differential & point swing — GW{gw}")

    fixture_status = cached_fixture_status(gw)
    for r in rows:
        team_id = NAME_TO_TEAM_ID.get(r["club"])
        r["_status"] = fixture_status.get(team_id, "unknown") if team_id else "unknown"

    counts = {s: sum(1 for r in rows if r["_status"] == s) for s in STATUS_ORDER}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Live", counts["live"])
    c2.metric("⏳ Upcoming", counts["upcoming"])
    c3.metric("✅ Finished", counts["finished"])
    if counts["unknown"]:
        c4.metric("❔ Unknown", counts["unknown"])

    status_filter = st.radio(
        "Filter by match status", ["All", "🔴 Live", "⏳ Upcoming", "✅ Finished"],
        horizontal=True, label_visibility="collapsed",
    )
    status_key = {"🔴 Live": "live", "⏳ Upcoming": "upcoming", "✅ Finished": "finished"}.get(status_filter)
    filtered_rows = rows if status_key is None else [r for r in rows if r["_status"] == status_key]

    if not filtered_rows:
        st.caption("No players in this status right now.")
    else:
        df_rows = pd.DataFrame(filtered_rows)[
            ["name", "position", "club", "_status", "A", "B", "diff", "live_pts", "point_swing"]
        ]
        df_rows.columns = ["Player", "Pos", "Club", "Status", "A", "B", "Diff", "GW Pts", "Swing"]
        df_rows["Status"] = df_rows["Status"].map(STATUS_LABEL)

        def _swing_style(v):
            if v > 0:
                return "color: #1a8754; font-weight: 600"
            if v < 0:
                return "color: #c0392b; font-weight: 600"
            return "color: #6b7280"

        st.dataframe(
            df_rows.style.map(_swing_style, subset=["Swing"]),
            hide_index=True, width="stretch", height=420,
        )
    st.caption(
        "Count: H2H captain + FPL captain = x4 · either alone = x2 · normal = x1 · "
        "Swing = diff × GW pts. Positive favours Team A, negative favours Team B. "
        "Status reflects each player's club fixture this gameweek."
    )

    swing_rows = [r for r in rows if r["point_swing"] != 0][:15]
    if swing_rows:
        st.markdown("###### Top swing players")
        names   = [r["name"] for r in swing_rows][::-1]
        swings  = [r["point_swing"] for r in swing_rows][::-1]
        colors_ = ["#1a8754" if s > 0 else "#0e7490" for s in swings][::-1]
        fig = go.Figure(go.Bar(x=swings, y=names, orientation="h", marker_color=colors_))
        fig.update_layout(height=max(300, 28 * len(names)), margin=dict(l=0, r=0, t=10, b=10),
                           xaxis_title="Point swing (Team A ►, ◄ Team B)")
        st.plotly_chart(fig, width="stretch")

    a_rows = [r for r in rows if r["point_swing"] > 0]
    b_rows = [r for r in rows if r["point_swing"] < 0]
    player_net = sum(r["point_swing"] for r in rows)
    hit_a = sum(mgr["transfer_hit"] for mgr in picks_a)
    hit_b = sum(mgr["transfer_hit"] for mgr in picks_b)
    net_h2h = player_net + (hit_a - hit_b)

    st.markdown("###### Swing summary")
    if a_rows:
        st.write(f"**{league_name_a} benefited from:** " +
                 ", ".join(f"{r['name']} (+{r['point_swing']} | {r['live_pts']}pts)" for r in a_rows))
    if b_rows:
        st.write(f"**{league_name_b} benefited from:** " +
                 ", ".join(f"{r['name']} ({r['point_swing']} | {r['live_pts']}pts)" for r in b_rows))
    net_label = (f"{league_name_a} +{player_net} pts" if player_net > 0 else
                 f"{league_name_b} +{abs(player_net)} pts" if player_net < 0 else "Balanced")
    net_h2h_label = (f"{league_name_a} +{net_h2h} pts" if net_h2h > 0 else
                     f"{league_name_b} +{abs(net_h2h)} pts" if net_h2h < 0 else "Level")
    st.write(f"**Net player points swing:** {net_label}")
    st.write(f"**Net H2H swing (incl. transfer hits):** {net_h2h_label}")


# ── Points race ────────────────────────────────────────────────────────────────
# ── GW Race (in-gameweek live race) ──────────────────────────────────────────
STAT_ICON = {
    "goals_scored": "⚽", "assists": "🅰", "clean_sheets": "🛡",
    "saves": "🧤", "penalties_saved": "🧤", "bonus": "⭐", "minutes": "⏱",
    "goals_conceded": "❌", "yellow_cards": "🟨", "red_cards": "🟥",
    "own_goals": "😬", "penalties_missed": "❌", "bps": "📊",
}
HEAVY_THRESHOLD = 10  # aggregate per-player swing (pts) to count as "heavy differential"

with tabs["GW Race"]:
    st.markdown(f"##### ⚡ GW{gw} — Live Points Race")
    st.caption(f"{league_name_a} vs {league_name_b} · every scoring event this gameweek, in order")

    fixture_ko = cached_fixture_kickoffs(gw)
    timeline = engine.build_intra_gw_timeline(
        picks_a, picks_b, player_map, res["live_explain"], live_scores, fixture_ko,
    )

    if not timeline:
        st.info(
            "No live scoring events yet for this gameweek. This fills in once matches "
            "kick off (or turn off **Skip live scores** in the sidebar if that's enabled)."
        )
    else:
        x       = list(range(len(timeline) + 1))
        sc_a    = [0] + [e["score_a_after"] for e in timeline]
        sc_b    = [0] + [e["score_b_after"] for e in timeline]
        lead    = [a - b for a, b in zip(sc_a, sc_b)]
        labels  = ["Kickoff"] + [
            f'{STAT_ICON.get(e["stat"], "")} {e["name"]} · {engine.STAT_LABELS.get(e["stat"], e["stat"])} '
            f'({"+" if e["raw_points"] >= 0 else ""}{e["raw_points"]}pts)'
            for e in timeline
        ]

        # Aggregate total swing per player across the whole GW so far
        player_swing = {}
        for idx, e in enumerate(timeline, start=1):
            pid = e["player_id"]
            slot = player_swing.setdefault(pid, {"name": e["name"], "swing": 0, "last_idx": idx})
            slot["swing"] += e["pts_swing_a"] - e["pts_swing_b"]
            slot["last_idx"] = idx
        heavy = {pid: v for pid, v in player_swing.items() if abs(v["swing"]) >= HEAVY_THRESHOLD}

        # ── Chart 1: lead margin, diverging fill (the "race") ────────────────
        lead_pos = [v if v > 0 else 0 for v in lead]
        lead_neg = [v if v < 0 else 0 for v in lead]
        fig = go.Figure()
        fig.add_scatter(x=x, y=lead_pos, mode="lines", line=dict(shape="hv", color="#1a8754", width=0),
                         fill="tozeroy", fillcolor="rgba(26,135,84,0.28)", name=league_name_a, hoverinfo="skip")
        fig.add_scatter(x=x, y=lead_neg, mode="lines", line=dict(shape="hv", color="#0e7490", width=0),
                         fill="tozeroy", fillcolor="rgba(14,116,144,0.28)", name=league_name_b, hoverinfo="skip")
        fig.add_scatter(x=x, y=lead, mode="lines+markers", line=dict(shape="hv", color="#e5e7eb", width=1.5),
                         marker=dict(size=4, color="#e5e7eb"), name="Lead", hovertext=labels,
                         hovertemplate="%{hovertext}<br><b>Lead: %{y:+d}</b><extra></extra>")

        for pid, v in heavy.items():
            idx = v["last_idx"]
            fig.add_scatter(
                x=[idx], y=[lead[idx]], mode="markers+text", showlegend=False,
                marker=dict(size=15, symbol="star", color="#f59e0b" if v["swing"] > 0 else "#f43f5e",
                            line=dict(width=1.5, color="white")),
                text=[f'{v["name"]} {v["swing"]:+d}'], textposition="top center",
                textfont=dict(size=11, color="#e5e7eb"),
                hovertemplate=f'<b>{v["name"]}</b><br>Total swing so far: {v["swing"]:+d} pts<extra></extra>',
            )

        fig.add_hline(y=0, line_dash="dot", line_color="#6b7280")
        fig.update_layout(
            height=420, margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(title="Scoring events (chronological)", showticklabels=False),
            yaxis_title=f"◀ {league_name_b}     Lead     {league_name_a} ▶",
            legend=dict(orientation="h", y=1.12, x=0),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

        # ── Chart 2: raw score race ────────────────────────────────────────────
        fig2 = go.Figure()
        fig2.add_scatter(x=x, y=sc_a, mode="lines+markers", line=dict(shape="hv", color="#1a8754", width=3),
                          marker=dict(size=4), name=league_name_a, hovertext=labels,
                          hovertemplate="%{hovertext}<br>%{y} pts<extra></extra>")
        fig2.add_scatter(x=x, y=sc_b, mode="lines+markers", line=dict(shape="hv", color="#0e7490", width=3),
                          marker=dict(size=4), name=league_name_b, hovertext=labels,
                          hovertemplate="%{hovertext}<br>%{y} pts<extra></extra>")
        fig2.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Scoring events (chronological)", showticklabels=False),
            yaxis_title="Cumulative GW points",
            legend=dict(orientation="h", y=1.12, x=0),
        )
        st.plotly_chart(fig2, width="stretch")

        # ── Heavy differential callouts ─────────────────────────────────────────
        st.markdown(f"###### 🌟 Heavy differentials this GW (swing ≥ {HEAVY_THRESHOLD} pts)")
        if heavy:
            for pid, v in sorted(heavy.items(), key=lambda kv: -abs(kv[1]["swing"])):
                side  = league_name_a if v["swing"] > 0 else league_name_b
                emoji = "🟢" if v["swing"] > 0 else "🔵"
                st.write(f'{emoji} **{v["name"]}** has swung **{abs(v["swing"])} pts** toward **{side}** so far')
        else:
            st.caption("No single player has swung this GW by 10+ points yet — tight race.")

        # ── Full chronological ticker ───────────────────────────────────────────
        with st.expander("Full event-by-event ticker"):
            df_evt = pd.DataFrame([{
                "Fixture":  f'{e.get("home","?")} vs {e.get("away","?")}',
                "Player":   e["name"],
                "Stat":     engine.STAT_LABELS.get(e["stat"], e["stat"]),
                "Pts":      e["raw_points"],
                f"{league_name_a} x": e["count_a"],
                f"{league_name_b} x": e["count_b"],
                "Score after": f'{e["score_a_after"]} – {e["score_b_after"]}',
            } for e in timeline])
            st.dataframe(df_evt, hide_index=True, width="stretch", height=420)


# ── Season Trend (past-gameweek analysis) ─────────────────────────────────────
with tabs["Season Trend"]:
    N = 8
    show_gws = list(range(max(1, gw - N + 1), gw + 1))

    def _cumulative(ids, histories, cap_idx):
        cum, running = [], 0
        for g in show_gws:
            gw_total = 0
            for i, mid in enumerate(ids):
                pts = histories.get(mid, {}).get("gw_scores", {}).get(g, {}).get("points", 0) or 0
                gw_total += pts * 2 if i == cap_idx else pts
            running += gw_total
            cum.append(running)
        return cum

    cum_a = _cumulative(res["team_a_ids"], histories_a, cap_a_idx)
    cum_b = _cumulative(res["team_b_ids"], histories_b, cap_b_idx)

    st.markdown(f"##### Cumulative points — last {len(show_gws)} GWs")
    fig = go.Figure()
    fig.add_scatter(x=[f"GW{g}" for g in show_gws], y=cum_a, mode="lines+markers",
                     name=league_name_a, line=dict(color="#1a8754", width=3))
    fig.add_scatter(x=[f"GW{g}" for g in show_gws], y=cum_b, mode="lines+markers",
                     name=league_name_b, line=dict(color="#0e7490", width=3))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=10), yaxis_title="Cumulative pts")
    st.plotly_chart(fig, width="stretch")

    lead = (cum_a[-1] if cum_a else 0) - (cum_b[-1] if cum_b else 0)
    if lead > 0:
        st.caption(f"{league_name_a} lead by {lead} pts over this window.")
    elif lead < 0:
        st.caption(f"{league_name_b} lead by {abs(lead)} pts over this window.")
    else:
        st.caption("Dead level over this window.")

    if len(show_gws) < 2:
        st.caption(
            "Only one gameweek of history is available so far this season — "
            "this chart fills out as more GWs are played."
        )


# ── Chip tracker ───────────────────────────────────────────────────────────────
if not settings["no_chips"]:
    with tabs["Chip Tracker"]:
        st.markdown("##### Chip tracker  (H1 = GW1-19, H2 = GW20-38)")
        current_half = 1 if gw in engine.CHIP_H1_GWS else 2
        for label, team_ids, managers, picks_list, chips_hist, league_nm in (
            ("A", res["team_a_ids"], managers_a, picks_a, chips_hist_a, league_name_a),
            ("B", res["team_b_ids"], managers_b, picks_b, chips_hist_b, league_name_b),
        ):
            st.markdown(f"**{league_nm} (Team {label})**")
            cols = ["Manager"]
            for c in engine.ALL_CHIPS:
                cols += [f"{engine.CHIP_DISPLAY[c]} H1", f"{engine.CHIP_DISPLAY[c]} H2"]
            data = []
            for i, (mid, pick) in enumerate(zip(team_ids, picks_list)):
                history = chips_hist.get(mid, [])
                active_chip = pick.get("active_chip")
                mgr_label = engine._mgr_label(managers, i, mid)
                row = [mgr_label]
                for c in engine.ALL_CHIPS:
                    for half in (1, 2):
                        entry = next((h for h in history if h["name"] == c and h["half"] == half), None)
                        if active_chip == c and current_half == half:
                            row.append("active")
                        elif entry:
                            row.append(f"used GW{entry['event']}")
                        else:
                            row.append("available")
                data.append(row)
            df = pd.DataFrame(data, columns=cols)

            def _chip_style(v):
                if v == "active":
                    return CHIP_STYLE["active"]
                if isinstance(v, str) and v.startswith("used"):
                    return CHIP_STYLE["used"]
                if v == "available":
                    return CHIP_STYLE["available"]
                return ""

            st.dataframe(
                df.style.map(_chip_style, subset=cols[1:]),
                hide_index=True, width="stretch",
            )
        st.caption("Every chip has two uses per season: once in GW1-19 (H1), once in GW20-38 (H2).")


# ── Squad summaries ────────────────────────────────────────────────────────────
if not settings["no_summary"]:
    with tabs["Squads"]:
        for label, team_ids, picks_list, cap_idx, managers, league_nm in (
            ("A", res["team_a_ids"], picks_a, cap_a_idx, managers_a, league_name_a),
            ("B", res["team_b_ids"], picks_b, cap_b_idx, managers_b, league_name_b),
        ):
            st.markdown(f"#### {league_nm} (Team {label})")
            for i, (mid, mgr) in enumerate(zip(team_ids, picks_list)):
                picks_data  = mgr["picks"]
                active_chip = mgr["active_chip"]
                eff_cap_id  = mgr.get("eff_cap_id")
                cap_mul     = mgr.get("cap_mul", engine.fpl_cap_multiplier(active_chip))
                mgr_label   = engine._mgr_label(managers, i, mid)
                cap_flag    = "  🅷 H2H CAPTAIN" if i == cap_idx else ""
                chip_flag   = f"  · {active_chip.upper()}" if active_chip else ""

                raw, hit, final = engine.manager_final_score(mgr, live_scores)
                with st.expander(f"Manager {i+1} — {mgr_label}{cap_flag}{chip_flag}  ·  {final} pts"):
                    starters = [p for p in picks_data if p["position"] <= 11]
                    bench    = [p for p in picks_data if p["position"] > 11]
                    ordered  = sorted(starters, key=lambda p: (
                        -(p["element"] == eff_cap_id), -p["is_vice_captain"]
                    ))

                    def _row(p, is_bench=False):
                        pl  = player_map.get(p["element"], {"name": f"#{p['element']}", "position": "?", "club": "?"})
                        pts = live_scores.get(p["element"], 0)
                        if is_bench:
                            note = "bboost" if active_chip == "bboost" else "bench"
                            mult = "x1"
                        else:
                            is_named_cap, is_eff_cap, is_vice = p["is_captain"], p["element"] == eff_cap_id, p["is_vice_captain"]
                            if is_named_cap and is_eff_cap:
                                note = "(c) x3" if cap_mul == 3 else "(c)"
                            elif is_eff_cap and not is_named_cap:
                                note = "(c*) auto-sub"
                            elif is_named_cap and not is_eff_cap:
                                note = "(c) blank — 0 mins"
                            elif is_vice:
                                note = "(v)"
                            else:
                                note = ""
                            mult = f"x{p['count']}"
                        return {"Pos": pl.get("position", "?"), "Player": pl["name"],
                                "Club": pl.get("club", "?"), "Pts": pts, "Mult": mult, "Note": note}

                    squad_recs = [_row(p) for p in ordered] + [_row(p, is_bench=True) for p in sorted(bench, key=lambda p: p["position"])]
                    st.dataframe(pd.DataFrame(squad_recs), hide_index=True, width="stretch")
                    hit_str = f"({hit})" if hit < 0 else "(0)"
                    st.caption(f"Score: {raw} {hit_str} = **{final}** pts")
