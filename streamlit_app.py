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
import math
import itertools
from collections import defaultdict
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fpl_h2h_v14 as engine

st.set_page_config(page_title="IML Scorecards", page_icon="⚽", layout="wide")

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
    "Chelsea":                  {"league_id": "1560363", "managers": ["Shashwat Prakash Dubey", "Amitash Srivastava", "Sourav Hemram", "Winayak Kumar"]},
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

# Maps the engine's numeric club id (engine.CLUBS) -> our TEAMS registry key.
# Needed because engine.CLUBS uses FPL's short club names ("Spurs", "Man City",
# "Nott'm Forest"...) while our roster uses full club names.
CLUB_ID_TO_TEAM = {
    1: "Arsenal", 2: "Aston Villa", 3: "Bournemouth", 4: "Brentford",
    5: "Brighton and Hove Albion", 6: "Chelsea", 7: "Coventry City",
    8: "Crystal Palace", 9: "Everton", 10: "Fulham", 11: "Hull City",
    12: "Ipswich Town", 13: "Leeds United", 14: "Liverpool",
    15: "Manchester City", 16: "Manchester United", 17: "Newcastle United",
    18: "Nottingham Forest", 19: "Tottenham Hotspurs", 20: "Sunderland",
}


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
def cached_raw_fixtures(gw: int):
    """One shared fetch of raw /fixtures/?event=<gw> data, reused for both the
    per-fixture tab list and the live/finished/upcoming status lookups."""
    try:
        return engine.fetch(f"{engine.BASE}/fixtures/?event={gw}")
    except Exception:
        return []


def _fixture_status(f: dict) -> str:
    """finished_provisional flips true right at full-time, well before
    'finished' (which waits on official bonus-point confirmation, sometimes
    ~1hr+ after the final whistle). Treating either as 'finished' is what
    actually matches reality — otherwise players from ended matches sit in
    'live' indefinitely and 'finished' stays empty."""
    if f.get("finished") or f.get("finished_provisional"):
        return "finished"
    if f.get("started"):
        return "live"
    return "upcoming"


def fixture_status_map(gw: int):
    """club_id -> 'finished' | 'live' | 'upcoming'."""
    status = {}
    for f in cached_raw_fixtures(gw):
        s = _fixture_status(f)
        for tid in (f.get("team_h"), f.get("team_a")):
            if tid:
                status[tid] = s
    return status


def gw_matchups(gw: int):
    """Every fixture this GW where both clubs map to one of our 20 registered
    teams, sorted by kickoff time. Each entry: team_a (home), team_b (away),
    kickoff (ISO str), status."""
    matchups = []
    for f in cached_raw_fixtures(gw):
        team_a_name = CLUB_ID_TO_TEAM.get(f.get("team_h"))
        team_b_name = CLUB_ID_TO_TEAM.get(f.get("team_a"))
        if not team_a_name or not team_b_name:
            continue
        matchups.append({
            "team_a": team_a_name, "team_b": team_b_name,
            "kickoff": f.get("kickoff_time") or "", "status": _fixture_status(f),
        })
    matchups.sort(key=lambda m: m["kickoff"] or "9999")
    return matchups


TEAM_TO_CLUB_ID = {v: k for k, v in CLUB_ID_TO_TEAM.items()}


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


# ── Shared constants used inside the per-fixture dashboard ────────────────────
STATUS_LABEL    = {"finished": "✅ FT", "live": "🔴 LIVE", "upcoming": "⏳ Upcoming", "unknown": "❔ —"}
STATUS_ORDER    = ["live", "upcoming", "finished", "unknown"]
NAME_TO_TEAM_ID = {v: k for k, v in engine.CLUBS.items()}  # engine club name -> engine club id

STAT_ICON = {
    "goals_scored": "⚽", "assists": "🅰", "clean_sheets": "🛡",
    "saves": "🧤", "penalties_saved": "🧤", "bonus": "⭐", "minutes": "⏱",
    "goals_conceded": "❌", "yellow_cards": "🟨", "red_cards": "🟥",
    "own_goals": "😬", "penalties_missed": "❌", "bps": "📊",
}
HEAVY_THRESHOLD = 10  # aggregate per-player swing (pts) to count as "heavy differential"


# ── Full analysis dashboard for one fixture (Scoreboard, GW Race, ...) ────────
def render_dashboard(res: dict, settings: dict, key_prefix: str) -> None:
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
    dash_tabs = dict(zip(tab_names, st.tabs(tab_names)))

    # ── Scoreboard ──────────────────────────────────────────────────────────
    with dash_tabs["Scoreboard"]:
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
                file_name=f"fpl_h2h_gw{gw}_{key_prefix}.pdf", mime="application/pdf",
                width="stretch", key=f"{key_prefix}_dl_pdf",
            )
        except SystemExit as e:
            dl_col1.warning(str(e))

        json_payload = json.dumps({
            "gw": gw, "league_a": league_name_a, "league_b": league_name_b,
            "total_a": total_a, "total_b": total_b, "differential": rows,
        }, indent=2)
        dl_col2.download_button(
            "Download JSON data", data=json_payload,
            file_name=f"fpl_h2h_gw{gw}_{key_prefix}.json", mime="application/json",
            width="stretch", key=f"{key_prefix}_dl_json",
        )

    # ── Team totals ─────────────────────────────────────────────────────────
    with dash_tabs["Team Totals"]:
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

    # ── Differential & swing ────────────────────────────────────────────────
    with dash_tabs["Differential & Swing"]:
        st.markdown(f"##### Ownership differential & point swing — GW{gw}")

        fixture_status = fixture_status_map(gw)
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
            horizontal=True, label_visibility="collapsed", key=f"{key_prefix}_status_filter",
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

    # ── GW Race (in-gameweek live race) ─────────────────────────────────────
    with dash_tabs["GW Race"]:
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

            player_swing = {}
            for idx, e in enumerate(timeline, start=1):
                pid = e["player_id"]
                slot = player_swing.setdefault(pid, {"name": e["name"], "swing": 0, "last_idx": idx})
                slot["swing"] += e["pts_swing_a"] - e["pts_swing_b"]
                slot["last_idx"] = idx
            heavy = {pid: v for pid, v in player_swing.items() if abs(v["swing"]) >= HEAVY_THRESHOLD}

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

            st.markdown(f"###### 🌟 Heavy differentials this GW (swing ≥ {HEAVY_THRESHOLD} pts)")
            if heavy:
                for pid, v in sorted(heavy.items(), key=lambda kv: -abs(kv[1]["swing"])):
                    side  = league_name_a if v["swing"] > 0 else league_name_b
                    emoji = "🟢" if v["swing"] > 0 else "🔵"
                    st.write(f'{emoji} **{v["name"]}** has swung **{abs(v["swing"])} pts** toward **{side}** so far')
            else:
                st.caption("No single player has swung this GW by 10+ points yet — tight race.")

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

    # ── Season Trend (past-gameweek analysis) ───────────────────────────────
    with dash_tabs["Season Trend"]:
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

    # ── Chip tracker ─────────────────────────────────────────────────────────
    if not settings["no_chips"]:
        with dash_tabs["Chip Tracker"]:
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

    # ── Squad summaries ──────────────────────────────────────────────────────
    if not settings["no_summary"]:
        with dash_tabs["Squads"]:
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


# ── Per-fixture tab: captain pickers, Analyse button, then the dashboard ──────
def render_fixture(m: dict, gw: int, no_live: bool, no_summary: bool, no_chips: bool) -> None:
    team_a_name, team_b_name = m["team_a"], m["team_b"]
    key_base = f"{gw}_{team_a_name}_{team_b_name}".replace(" ", "_")

    status_label = {"finished": "✅ Full time", "live": "🔴 Live", "upcoming": "⏳ Not started"}[m["status"]]
    ko = m["kickoff"]
    ko_display = f"{ko[:10]} {ko[11:16]} UTC" if len(ko) >= 16 else "Kickoff TBC"
    st.caption(f"{ko_display} · {status_label}")

    col1, col2 = st.columns(2)
    cap_a = col1.selectbox(f"{team_a_name} captain", TEAMS[team_a_name]["managers"], key=f"capA_{key_base}")
    cap_b = col2.selectbox(f"{team_b_name} captain", TEAMS[team_b_name]["managers"], key=f"capB_{key_base}")

    analyse_clicked = st.button(
        f"Analyse {team_a_name} vs {team_b_name}", key=f"btn_{key_base}", type="primary",
    )

    result_key = f"result_{key_base}"
    if analyse_clicked:
        try:
            with st.spinner("Talking to the FPL API — leagues, picks, chip history..."):
                fetched = run_pipeline(
                    gw, TEAMS[team_a_name]["league_id"], TEAMS[team_b_name]["league_id"],
                    cap_a, cap_b, no_live,
                )
            st.session_state[result_key] = dict(
                res=fetched, settings=dict(no_summary=no_summary, no_chips=no_chips),
            )
            st.toast(f"{team_a_name} vs {team_b_name} analysed!", icon="✅")
        except SystemExit as e:
            st.error(str(e).strip() or "Couldn't resolve one of the captains — check the roster and try again.")
        except Exception as e:
            st.error(f"Something went wrong fetching data: {e}")

    stored = st.session_state.get(result_key)
    if stored is None:
        st.info("Pick both captains above, then click **Analyse** to load this fixture.")
        return

    st.divider()
    render_dashboard(stored["res"], stored["settings"], key_base)


# ── Weekly Matchups mode: one tab per this-gameweek fixture ──────────────────
def render_weekly_matchups(gw: int, no_live: bool, no_summary: bool, no_chips: bool) -> None:
    try:
        with st.spinner(f"Loading GW{gw} fixtures..."):
            matchups = gw_matchups(gw)
    except Exception as e:
        st.error(f"Couldn't load fixtures for this gameweek: {e}")
        return

    if not matchups:
        st.info(
            f"No fixtures found for GW{gw} yet — could be a blank gameweek, or fixtures "
            "haven't been released. Try a different gameweek in the sidebar."
        )
        return

    fixture_labels = [f'{m["team_a"]} vs {m["team_b"]}' for m in matchups]
    fixture_tabs = st.tabs(fixture_labels)

    for matchup, tab in zip(matchups, fixture_tabs):
        with tab:
            render_fixture(matchup, gw, no_live, no_summary, no_chips)



# ── League-wide analytics engine ────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def cached_iml_registry():
    """Return the 80 IML members grouped by their 20 club-named H2H teams."""
    out = []
    for team_name, cfg in TEAMS.items():
        managers = cached_league_managers(cfg["league_id"])
        for m in managers:
            out.append({
                "team": team_name,
                "league_id": cfg["league_id"],
                "manager_id": str(m["id"]),
                "manager": m.get("manager") or m.get("name") or str(m["id"]),
            })
    # De-duplicate defensively in case a manager appears in two registered leagues.
    seen = set()
    deduped = []
    for row in out:
        if row["manager_id"] not in seen:
            seen.add(row["manager_id"])
            deduped.append(row)
    return tuple(deduped)


@st.cache_data(ttl=180, show_spinner=False)
def cached_all_current_picks(gw: int, manager_ids: tuple):
    """Fetch current-gameweek picks for all IML managers."""
    out = {}
    for mid in manager_ids:
        try:
            out[mid] = engine.get_picks(mid, gw)
        except Exception:
            out[mid] = {}
    return out


@st.cache_data(ttl=600, show_spinner=False)
def cached_player_summaries(player_ids: tuple):
    """Fetch element-summary histories for the supplied player IDs."""
    out = {}
    for pid in player_ids:
        try:
            out[int(pid)] = engine.fetch(f"{engine.BASE}/element-summary/{int(pid)}/")
        except Exception:
            out[int(pid)] = {}
    return out


@st.cache_data(ttl=600, show_spinner=False)
def cached_fixtures_all():
    """Fetch the full fixture list once; analytics uses it for tickers and projections."""
    try:
        return engine.fetch(f"{engine.BASE}/fixtures/")
    except Exception:
        return []


@st.cache_data(ttl=900, show_spinner=False)
def cached_all_manager_histories(manager_ids: tuple):
    return engine.get_all_season_histories(list(manager_ids))


@st.cache_data(ttl=900, show_spinner=False)
def cached_manager_transfers(manager_id: str):
    try:
        return engine.fetch(f"{engine.BASE}/entry/{manager_id}/transfers/")
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def cached_manager_gw_picks(manager_id: str, gw: int):
    try:
        return engine.get_picks(manager_id, gw)
    except Exception:
        return {}


def current_or_next_gw(bootstrap: dict) -> int:
    for ev in bootstrap.get("events", []):
        if ev.get("is_current"):
            return int(ev["id"])
    for ev in bootstrap.get("events", []):
        if ev.get("is_next"):
            return int(ev["id"])
    for ev in bootstrap.get("events", []):
        if not ev.get("finished"):
            return int(ev["id"])
    return 1


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _pct_rank(values):
    """Simple percentile rank, robust to a short league at season start."""
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    return {
        v: (sum(x <= v for x in ordered) - 1) / max(1, n - 1)
        for v in set(values)
    }


def _normalize01(v, lo, hi):
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def _difficulty_factor(fdr):
    # FPL FDR is 1 (best) → 5 (worst). 3 is the neutral baseline.
    return {1: 1.16, 2: 1.08, 3: 1.00, 4: 0.92, 5: 0.84}.get(int(fdr or 3), 1.0)


def _defcon_threshold(position):
    return {"DEF": 10, "MID": 12, "FWD": 12}.get(position, 999)


def _player_position_map(bootstrap):
    return {int(e["id"]): engine.POSITIONS.get(e.get("element_type"), "?")
            for e in bootstrap.get("elements", [])}


def _team_name_map(bootstrap):
    return {int(t["id"]): t.get("name") or str(t["id"]) for t in bootstrap.get("teams", [])}


def _fixture_index(fixtures):
    idx = defaultdict(list)
    for f in fixtures:
        event = f.get("event")
        if not event:
            continue
        h = int(f.get("team_h") or 0)
        a = int(f.get("team_a") or 0)
        if h:
            idx[h].append({
                "event": int(event), "opponent_id": a, "home": True,
                "kickoff": f.get("kickoff_time") or "",
                "fdr": int(f.get("team_h_difficulty") or 3),
                "fixture_id": f.get("id"),
                "started": bool(f.get("started")),
                "finished": bool(f.get("finished") or f.get("finished_provisional")),
            })
        if a:
            idx[a].append({
                "event": int(event), "opponent_id": h, "home": False,
                "kickoff": f.get("kickoff_time") or "",
                "fdr": int(f.get("team_a_difficulty") or 3),
                "fixture_id": f.get("id"),
                "started": bool(f.get("started")),
                "finished": bool(f.get("finished") or f.get("finished_provisional")),
            })
    for k in idx:
        idx[k].sort(key=lambda x: (x["event"], x["kickoff"]))
    return idx


def build_player_analytics(bootstrap, fixtures, player_summaries=None, horizon=5, start_gw=None):
    """
    Transparent projection model:
      1) Minutes security: start rate + average minutes.
      2) Underlying: xG90 + xA90 + DC90.
      3) Recent form: recent FPL points adjusted by fixture difficulty.
      4) Next-fixture multiplier from official FPL FDR.
      5) Projected points = blended season/recent baseline × minutes security × FDR.
    """
    player_summaries = player_summaries or {}
    teams = _team_name_map(bootstrap)
    fx_idx = _fixture_index(fixtures)
    current_gw = int(start_gw or current_or_next_gw(bootstrap))
    rows = []

    for p in bootstrap.get("elements", []):
        pid = int(p["id"])
        pos = engine.POSITIONS.get(p.get("element_type"), "?")
        hist = (player_summaries.get(pid) or {}).get("history", [])

        # Use recent player history when available; bootstrap remains the fallback.
        recent = sorted(hist, key=lambda h: int(h.get("round") or 0))[-5:]
        played = [h for h in recent if _safe_float(h.get("minutes")) > 0]
        minutes_total = sum(_safe_float(h.get("minutes")) for h in hist)
        starts_total = sum(_safe_float(h.get("starts")) for h in hist)
        appearances = len([h for h in hist if _safe_float(h.get("minutes")) > 0])

        avg_minutes = (minutes_total / appearances) if appearances else _safe_float(p.get("minutes"), 0) / max(1, _safe_float(p.get("games_played"), 1))
        start_rate = (starts_total / max(1, appearances)) if hist else 0.75 if _safe_float(p.get("minutes")) >= 60 else 0.4
        minutes_security = max(0.0, min(1.0, 0.58 * min(avg_minutes / 90.0, 1.0) + 0.42 * min(start_rate, 1.0)))

        xg = sum(_safe_float(h.get("expected_goals")) for h in hist)
        xa = sum(_safe_float(h.get("expected_assists")) for h in hist)
        mins = max(1.0, minutes_total)
        xg90 = 90.0 * xg / mins
        xa90 = 90.0 * xa / mins
        xgi90 = xg90 + xa90

        dc_total = sum(_safe_float(h.get("defensive_contribution")) for h in hist)
        dc90 = 90.0 * dc_total / mins if dc_total else 0.0
        threshold = _defcon_threshold(pos)
        defcon_hits = [h for h in played if _safe_float(h.get("defensive_contribution")) >= threshold]
        defcon_reliability = len(defcon_hits) / len(played) if played else 0.0

        # Bootstrap fallback for early season / players without detailed history.
        season_ppg = _safe_float(p.get("points_per_game"), 0.0)
        season_form = _safe_float(p.get("form"), season_ppg)
        if played:
            recent_ppg = sum(_safe_float(h.get("total_points")) for h in played) / len(played)
        else:
            recent_ppg = season_form

        # Fixture-adjusted recent form: punish hard recent schedules, reward easy ones.
        recent_adj = []
        for h in played:
            fdr = 3
            for f in fx_idx.get(int(p.get("team") or 0), []):
                if int(f["event"]) == int(h.get("round") or -1):
                    fdr = f["fdr"]
                    break
            recent_adj.append(_safe_float(h.get("total_points")) * _difficulty_factor(fdr))
        fixture_adjusted_form = sum(recent_adj) / len(recent_adj) if recent_adj else season_form

        # Forward fixture ticker.
        fixtures_next = [
            f for f in fx_idx.get(int(p.get("team") or 0), [])
            if f["event"] >= current_gw and not f["finished"]
        ][:horizon]

        proj = []
        for f in fixtures_next:
            fixture_baseline = (
                0.55 * season_ppg +
                0.25 * recent_ppg +
                0.20 * fixture_adjusted_form
            )
            underlying_signal = 1.55 * xgi90 + (0.10 * dc90)
            minutes_factor = 0.62 + 0.38 * minutes_security
            match_projection = (fixture_baseline + underlying_signal) * minutes_factor * _difficulty_factor(f["fdr"])
            # Small role correction for goalkeepers/defenders through clean sheets/defcon.
            if pos in ("GKP", "DEF"):
                match_projection += 0.18 * defcon_reliability
            proj.append(match_projection)

        projected_next = sum(proj)
        next_ppg = projected_next / len(proj) if proj else max(0.0, season_ppg)

        # Simple reliability blend, constrained to an intuitive 0-100 score.
        ppg_stability = 1.0
        if len(played) >= 2:
            vals = [_safe_float(h.get("total_points")) for h in played]
            mean = sum(vals) / len(vals)
            stdev = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
            ppg_stability = 1.0 / (1.0 + (stdev / max(1.0, mean)))
        reliability = 100.0 * (0.58 * minutes_security + 0.24 * defcon_reliability + 0.18 * ppg_stability)

        rows.append({
            "id": pid,
            "name": p.get("web_name") or f"#{pid}",
            "full_name": f'{p.get("first_name","")} {p.get("second_name","")}'.strip(),
            "position": pos,
            "club": teams.get(int(p.get("team") or 0), "?"),
            "club_id": int(p.get("team") or 0),
            "price": _safe_float(p.get("now_cost")) / 10.0,
            "ownership": _safe_float(p.get("selected_by_percent")),
            "season_points": _safe_float(p.get("total_points")),
            "season_ppg": season_ppg,
            "form": season_form,
            "minutes": _safe_float(p.get("minutes")),
            "minutes_security": minutes_security * 100.0,
            "xg90": xg90,
            "xa90": xa90,
            "xgi90": xgi90,
            "dc90": dc90,
            "defcon_reliability": defcon_reliability * 100.0,
            "fixture_adjusted_form": fixture_adjusted_form,
            "projected_next": projected_next,
            "projected_ppg": next_ppg,
            "reliability": reliability,
            "fixtures_next": fixtures_next,
        })
    return rows


def _squad_player_ids(all_picks):
    ids = set()
    for data in all_picks.values():
        for p in data.get("picks", []) if data else []:
            try:
                ids.add(int(p["element"]))
            except Exception:
                pass
    return tuple(sorted(ids))


def league_ownership_table(registry, all_picks, analytics_rows):
    analytics_map = {r["id"]: r for r in analytics_rows}
    owners = defaultdict(set)
    captainters = defaultdict(float)
    team_counts = defaultdict(lambda: defaultdict(int))

    for member in registry:
        mid = member["manager_id"]
        data = all_picks.get(mid) or {}
        for p in data.get("picks", []):
            pid = int(p["element"])
            owners[pid].add(mid)
            if p.get("position", 0) <= 11:
                team_counts[member["team"]][pid] += 1
                if p.get("is_captain"):
                    # Captain ownership counts as one extra unit; TC gets two extra.
                    mult = 2 if data.get("active_chip") == "3xc" else 1
                    captainters[pid] += mult

    n = max(1, len(registry))
    rows = []
    for pid, owner_set in owners.items():
        r = analytics_map.get(pid)
        if not r:
            continue
        ownership_pct = 100.0 * len(owner_set) / n
        eo_pct = 100.0 * (len(owner_set) + captainters.get(pid, 0)) / n
        team_vals = [team_counts[t].get(pid, 0) for t in TEAMS.keys()]
        max_diff = max(team_vals) - min(team_vals) if team_vals else 0
        rows.append({
            **r,
            "iml_owned": len(owner_set),
            "ownership_pct": ownership_pct,
            "captain_pct": 100.0 * captainters.get(pid, 0) / n,
            "effective_ownership_pct": eo_pct,
            "max_team_ownership_diff": max_diff,
            "swing_potential": max_diff * r["projected_ppg"],
        })
    rows.sort(key=lambda x: (-x["ownership_pct"], -x["projected_next"]))
    return rows, team_counts


def build_iml11(ownership_rows, ownership_threshold=20.0):
    eligible = [r for r in ownership_rows if r["ownership_pct"] >= ownership_threshold]
    if not eligible:
        return []

    # IML Template XI score: ownership is the floor (protect rank),
    # projection is the upside, and security/reliability stop fragile picks winning.
    for r in eligible:
        r["iml11_score"] = (
            0.30 * min(100.0, r["ownership_pct"]) +
            0.28 * min(100.0, 20.0 * r["projected_ppg"]) +
            0.15 * r["minutes_security"] +
            0.12 * min(100.0, r["fixture_adjusted_form"] * 10.0) +
            0.10 * r["defcon_reliability"] +
            0.05 * r["reliability"]
        )

    # Greedy construction with FPL formation + max-three-per-club constraints.
    selected = []
    counts = defaultdict(int)

    def pick_best(position=None, minimum=False):
        cand = [x for x in eligible if (position is None or x["position"] == position)
                and x["id"] not in {s["id"] for s in selected}
                and counts[x["club_id"]] < 3]
        cand.sort(key=lambda x: x["iml11_score"], reverse=True)
        if not cand:
            return None
        x = cand[0]
        selected.append(x)
        counts[x["club_id"]] += 1
        return x

    pick_best("GKP")
    for _ in range(3):
        pick_best("DEF")
    for _ in range(2):
        pick_best("MID")
    pick_best("FWD")

    while len(selected) < 11:
        # Respect 5 defenders max, 5 mids max, 3 forwards max.
        pos_counts = {p: sum(1 for s in selected if s["position"] == p) for p in ("DEF", "MID", "FWD", "GKP")}
        cand = [x for x in eligible
                if x["id"] not in {s["id"] for s in selected}
                and counts[x["club_id"]] < 3
                and (
                    x["position"] == "DEF" and pos_counts["DEF"] < 5 or
                    x["position"] == "MID" and pos_counts["MID"] < 5 or
                    x["position"] == "FWD" and pos_counts["FWD"] < 3
                )]
        if not cand:
            break
        cand.sort(key=lambda x: x["iml11_score"], reverse=True)
        x = cand[0]
        selected.append(x)
        counts[x["club_id"]] += 1

    return selected[:11]


def h2h_live_win_probability(picks_a, picks_b, player_map, live_scores, analytics_map,
                             manager_histories=None, simulations=1800):
    """
    Monte Carlo forecast of a live H2H race.
    Locked/live points are fixed; only players whose fixtures are not finished
    receive a stochastic remainder based on projected points and reliability.
    """
    def starter_map(picks):
        out = defaultdict(int)
        for mgr in picks:
            chip = mgr.get("active_chip")
            for p in mgr.get("picks", []):
                if chip == "bboost" or p.get("position", 0) <= 11:
                    out[p["element"]] += p.get("count", 1)
        return out

    ca = starter_map(picks_a)
    cb = starter_map(picks_b)
    ids = set(ca) | set(cb)

    fixed_a = sum(ca[pid] * live_scores.get(pid, 0) for pid in ids)
    fixed_b = sum(cb[pid] * live_scores.get(pid, 0) for pid in ids)

    future = []
    for pid in ids:
        r = analytics_map.get(pid)
        if not r:
            continue
        # Current live score is locked; projected remainder is the difference
        # between next-GW projection and already scored points, floored at zero.
        rem_mean = max(0.0, r.get("projected_ppg", 0.0) - live_scores.get(pid, 0))
        if rem_mean <= 0:
            continue
        reliability = max(0.15, min(0.92, r.get("reliability", 50.0) / 100.0))
        future.append((pid, ca.get(pid, 0), cb.get(pid, 0), rem_mean, reliability))

    # Deterministic seed from the current live state gives stable UI values within a run.
    import random
    seed = int(sum((pid + 1) * (live_scores.get(pid, 0) + 7) for pid in ids)) % (2**32 - 1)
    rng = random.Random(seed)
    wins_a = wins_b = draws = 0
    for _ in range(simulations):
        sa, sb = fixed_a, fixed_b
        for pid, wa, wb, mean, rel in future:
            # Gamma-like positive distribution, concentrated around mean.
            shape = max(1.2, 1.5 + 3.0 * rel)
            scale = mean / shape
            sample = rng.gammavariate(shape, scale)
            sa += wa * sample
            sb += wb * sample
        if sa > sb:
            wins_a += 1
        elif sb > sa:
            wins_b += 1
        else:
            draws += 1
    denom = max(1, simulations)
    return (100.0 * wins_a / denom, 100.0 * draws / denom, 100.0 * wins_b / denom, future)


def simulate_manager_score(mgr_picks, gw, player_histories, player_map):
    """Return raw GW score and simple bench/captain regret quantities."""
    picks = mgr_picks.get("picks", [])
    active_chip = mgr_picks.get("active_chip")
    by_id = {int(p["element"]): p for p in picks}

    def stat(pid):
        hist = (player_histories.get(pid) or {}).get("history", [])
        for h in hist:
            if int(h.get("round") or -1) == int(gw):
                return h
        return {}

    # Actual effective captain.
    actual_cap = next((p for p in picks if p.get("is_captain")), None)
    vice = next((p for p in picks if p.get("is_vice_captain")), None)
    if actual_cap and _safe_float(stat(int(actual_cap["element"])).get("minutes")) == 0 and active_chip != "bboost" and vice:
        actual_cap = vice
    cap_pid = int(actual_cap["element"]) if actual_cap else None
    cap_mult = 3 if active_chip == "3xc" else 2

    starters = [p for p in picks if p.get("position", 0) <= 11]
    bench = [p for p in picks if p.get("position", 0) > 11]
    raw = 0.0
    for p in starters:
        raw += _safe_float(stat(int(p["element"])).get("total_points"))
    if cap_pid:
        raw += _safe_float(stat(cap_pid).get("total_points")) * (cap_mult - 1)
    if active_chip == "bboost":
        raw += sum(_safe_float(stat(int(p["element"])).get("total_points")) for p in bench)

    starter_pts = {int(p["element"]): _safe_float(stat(int(p["element"])).get("total_points")) for p in starters}
    best_cap = max(starter_pts.values()) if starter_pts else 0.0
    actual_cap_pts = starter_pts.get(cap_pid, 0.0)
    captain_regret = max(0.0, (best_cap - actual_cap_pts) * cap_mult)

    # Bench waste = bench points that were not consumed by a no-minutes auto-sub.
    bench_points = [_safe_float(stat(int(p["element"])).get("total_points")) for p in bench]
    bench_waste = sum(bench_points)
    if active_chip != "bboost":
        # Approximate auto-sub consumption: every zero-minute starter can consume the
        # first eligible bench player while maintaining 1 GK / 3 DEF / 2 MID+FWD.
        remaining_bench = list(bench)
        for sp in starters:
            if _safe_float(stat(int(sp["element"])).get("minutes")) > 0:
                continue
            for bp in list(remaining_bench):
                bpos = int(by_id.get(int(bp["element"]), {}).get("position", 0))
                spos = int(sp.get("position", 0))
                # Basic FPL positional legality check.
                starter_defs = sum(1 for x in starters if int(x.get("position", 0)) == 2 and x is not sp)
                starter_gk = sum(1 for x in starters if int(x.get("position", 0)) == 1 and x is not sp)
                # GK can only replace GK; outfield bench cannot be GK.
                legal = (spos == 1 and bpos == 1) or (
                    spos != 1 and bpos != 1 and
                    starter_defs + (1 if bpos == 2 else 0) >= 3 and
                    (starter_gk + 1) >= 1
                )
                if legal and _safe_float(stat(int(bp["element"])).get("minutes")) > 0:
                    bench_waste -= _safe_float(stat(int(bp["element"])).get("total_points"))
                    remaining_bench.remove(bp)
                    break

    return raw, max(0.0, bench_waste), captain_regret


def manager_transfer_roi(manager_id, transfers, player_histories, window=4):
    """4-GW transfer ROI: incoming points minus outgoing points over next N GWs, less hit."""
    rows = []
    for t in transfers or []:
        try:
            gw = int(t.get("event"))
            inn = int(t.get("element_in"))
            out = int(t.get("element_out"))
        except Exception:
            continue
        h_in = (player_histories.get(inn) or {}).get("history", [])
        h_out = (player_histories.get(out) or {}).get("history", [])
        end = gw + window - 1
        in_pts = sum(_safe_float(h.get("total_points")) for h in h_in if gw <= int(h.get("round") or -1) <= end)
        out_pts = sum(_safe_float(h.get("total_points")) for h in h_out if gw <= int(h.get("round") or -1) <= end)
        cost = abs(_safe_float(t.get("cost"), 0.0))
        rows.append({
            "gw": gw, "incoming": inn, "outgoing": out,
            "incoming_pts": in_pts, "outgoing_pts": out_pts,
            "hit": cost, "roi": in_pts - out_pts - cost,
        })
    return rows


def manager_power_rankings(registry, histories, chip_histories):
    records = []
    for member in registry:
        mid = member["manager_id"]
        h = histories.get(mid, {})
        gws = [v for g, v in sorted(h.get("gw_scores", {}).items()) if v.get("points") is not None]
        pts = [float(v.get("points") or 0) for v in gws]
        total = float(gws[-1].get("total_points") or 0) if gws else 0.0
        recent = pts[-5:] if pts else []
        mean = sum(pts) / len(pts) if pts else 0.0
        stdev = (sum((x - mean) ** 2 for x in pts) / len(pts)) ** 0.5 if pts else 0.0
        consistency = 100.0 / (1.0 + stdev / max(1.0, mean))
        recent_mean = sum(recent) / len(recent) if recent else mean

        chips = chip_histories.get(mid, [])
        chip_events = [int(c["event"]) for c in chips]
        chip_roi = 0.0
        if chips and pts:
            all_gw_nums = sorted(h.get("gw_scores", {}).keys())
            nonchip = [
                float(h.get("gw_scores", {}).get(g, {}).get("points", 0) or 0)
                for g in all_gw_nums
                if int(g) not in chip_events
            ]
            baseline = sum(nonchip) / len(nonchip) if nonchip else mean
            chip_pts = sum(
                h.get("gw_scores", {}).get(g, {}).get("points", 0) or 0
                for g in chip_events
            )
            chip_roi = chip_pts / len(chips) - baseline

        records.append({
            **member,
            "total_points": total,
            "recent_mean": recent_mean,
            "consistency": consistency,
            "chip_roi": chip_roi,
            "weeks": len(pts),
        })

    if not records:
        return []

    p_total = _pct_rank([r["total_points"] for r in records])
    p_recent = _pct_rank([r["recent_mean"] for r in records])
    p_cons = _pct_rank([r["consistency"] for r in records])
    p_chip = _pct_rank([r["chip_roi"] for r in records])

    for r in records:
        r["power_score"] = 100.0 * (
            0.50 * p_total[r["total_points"]] +
            0.25 * p_recent[r["recent_mean"]] +
            0.20 * p_cons[r["consistency"]] +
            0.05 * p_chip[r["chip_roi"]]
        )
    records.sort(key=lambda x: (-x["power_score"], -x["total_points"]))
    for i, r in enumerate(records, 1):
        r["rank"] = i
    return records


def differential_finder(ownership_rows, min_ownership=2.5):
    """Low-owned player shortlist using projection, reliability, fixture and crowd leverage."""
    rows = []
    for r in ownership_rows:
        own = r["ownership_pct"]
        if own > min_ownership + 7.5:
            continue
        upside = 0.55 * r["projected_ppg"] + 0.20 * r["fixture_adjusted_form"] + 0.15 * r["reliability"] / 20.0 + 0.10 * r["minutes_security"] / 20.0
        leverage = max(0.0, 100.0 - own) / 100.0
        score = upside * (0.65 + 0.35 * leverage)
        rows.append({**r, "differential_score": score})
    rows.sort(key=lambda x: (-x["differential_score"], -x["projected_ppg"]))
    return rows[:30]


def member_snapshot(manager_id, gw, player_histories, player_map):
    """Historical manager lab: captain regret, bench waste, consistency and season totals."""
    weekly = []
    for g in range(1, gw + 1):
        picks = cached_manager_gw_picks(manager_id, g)
        if not picks or not picks.get("picks"):
            continue
        raw, bench_waste, cap_regret = simulate_manager_score(
            picks, g, player_histories, player_map
        )
        weekly.append({"GW": g, "Points": raw, "Bench Wasted": bench_waste, "Captain Regret": cap_regret})

    pts = [x["Points"] for x in weekly]
    mean = sum(pts) / len(pts) if pts else 0.0
    sd = (sum((x - mean) ** 2 for x in pts) / len(pts)) ** 0.5 if pts else 0.0
    consistency = 100.0 / (1.0 + sd / max(1.0, mean))
    return {
        "weekly": weekly,
        "avg_gw": mean,
        "consistency": consistency,
        "bench_wasted": sum(x["Bench Wasted"] for x in weekly),
        "captain_regret": sum(x["Captain Regret"] for x in weekly),
    }


def format_fixture_ticker(row, team_map):
    ticker = []
    for f in row.get("fixtures_next", []):
        opp = team_map.get(f["opponent_id"], str(f["opponent_id"]))
        ticker.append({
            "GW": f["event"],
            "Fixture": f'{"H" if f["home"] else "A"} · {opp}',
            "FDR": f["fdr"],
            "Proj": round((row.get("projected_ppg") or 0.0) * _difficulty_factor(f["fdr"]), 2),
        })
    return ticker


# ── Sidebar: global settings ────────────────────────────────────────────────────
for _k, _default in (("gw_value", 1), ("skip_live_value", False),
                     ("skip_summary_value", False), ("skip_chips_value", False)):
    if _k not in st.session_state:
        st.session_state[_k] = _default

with st.sidebar:
    st.header("⚽ IML Scorecards")
    mode = st.radio("Mode", ["League Analytics", "This Week's Matchups"], key="app_mode")

    bootstrap_sidebar = cached_bootstrap()
    default_gw = current_or_next_gw(bootstrap_sidebar)

    st.divider()
    if mode == "This Week's Matchups":
        st.header("Gameweek")
        gw = st.number_input("Gameweek", min_value=1, max_value=38,
                             value=int(st.session_state.gw_value or default_gw),
                             step=1, key="gw_input")
        st.session_state.gw_value = gw

        no_live = st.checkbox(
            "Skip live scores",
            value=st.session_state.skip_live_value,
            key="skip_live",
            help="Use for a finished season or faster testing.",
        )
        st.session_state.skip_live_value = no_live
        no_summary = st.checkbox(
            "Skip squad summaries",
            value=st.session_state.skip_summary_value,
            key="skip_summary",
        )
        st.session_state.skip_summary_value = no_summary
        no_chips = st.checkbox(
            "Skip chip tracker",
            value=st.session_state.skip_chips_value,
            key="skip_chips",
        )
        st.session_state.skip_chips_value = no_chips
    else:
        st.header("Analytics")
        gw = st.number_input(
            "Analysis gameweek",
            min_value=1, max_value=38,
            value=int(st.session_state.gw_value or default_gw),
            step=1, key="analytics_gw",
        )
        st.session_state.gw_value = gw
        horizon = st.slider("Projection horizon (GWs)", 3, 8, 5, key="analytics_horizon")
        ownership_threshold = st.slider(
            "Highly-owned threshold (%)", 5.0, 40.0, 20.0, 2.5,
            key="ownership_threshold",
        )
        st.caption(
            "League Analytics provides forward-looking player, squad, manager and IML-wide analysis."
        )

# ── League analytics UI ─────────────────────────────────────────────────────────
def _pre_gw_player_metric(pid, history, cutoff_gw):
    """Build lightweight pre-GW metrics using history strictly through cutoff_gw."""
    hist = [h for h in (history or []) if int(h.get("round") or 0) <= int(cutoff_gw)]
    hist.sort(key=lambda h: int(h.get("round") or 0))
    played = [h for h in hist if _safe_float(h.get("minutes")) > 0]
    mins = sum(_safe_float(h.get("minutes")) for h in hist)
    apps = len(played)
    starts = sum(_safe_float(h.get("starts")) for h in hist)
    avg_minutes = mins / apps if apps else 0.0
    start_rate = starts / apps if apps else 0.0
    min_sec = max(0.0, min(100.0, 100.0 * (0.62 * min(avg_minutes / 90.0, 1.0) + 0.38 * min(start_rate, 1.0))))

    xg = sum(_safe_float(h.get("expected_goals")) for h in hist)
    xa = sum(_safe_float(h.get("expected_assists")) for h in hist)
    xg90 = 90.0 * xg / max(1.0, mins)
    xa90 = 90.0 * xa / max(1.0, mins)
    xgi90 = xg90 + xa90

    dc_values = [_safe_float(h.get("defensive_contribution")) for h in played]
    dc90 = 90.0 * sum(dc_values) / max(1.0, mins)
    pos = None
    # position is supplied by caller later; use threshold in caller if needed.

    recent = played[-5:]
    recent_ppg = sum(_safe_float(h.get("total_points")) for h in recent) / len(recent) if recent else 0.0
    form_vals = [ _safe_float(h.get("total_points")) for h in recent ]
    mean = sum(form_vals) / len(form_vals) if form_vals else 0.0
    stdev = (sum((x-mean)**2 for x in form_vals)/len(form_vals))**0.5 if form_vals else 0.0
    stability = 1.0 / (1.0 + stdev / max(1.0, mean)) if form_vals else 0.5

    ict_vals = [_safe_float(h.get("ict_index")) for h in recent if h.get("ict_index") not in (None, "")]
    inf_vals = [_safe_float(h.get("influence")) for h in recent if h.get("influence") not in (None, "")]
    cre_vals = [_safe_float(h.get("creativity")) for h in recent if h.get("creativity") not in (None, "")]
    thr_vals = [_safe_float(h.get("threat")) for h in recent if h.get("threat") not in (None, "")]

    return {
        "history": hist,
        "minutes_security": min_sec,
        "xg90": xg90,
        "xa90": xa90,
        "xgi90": xgi90,
        "dc90": dc90,
        "recent_ppg": recent_ppg,
        "stability": stability,
        "ict": sum(ict_vals)/len(ict_vals) if ict_vals else 0.0,
        "influence": sum(inf_vals)/len(inf_vals) if inf_vals else 0.0,
        "creativity": sum(cre_vals)/len(cre_vals) if cre_vals else 0.0,
        "threat": sum(thr_vals)/len(thr_vals) if thr_vals else 0.0,
    }


def _build_simple_scout_rows(team_a, team_b, gw, horizon=1):
    """Fast pre-GW differential board: only two 4-manager squads + their players."""
    bootstrap = cached_bootstrap()
    player_map = cached_player_map(bootstrap)
    cutoff = max(0, int(gw) - 1)
    fixture_data = cached_fixtures_all()
    teams = _team_name_map(bootstrap)

    ma = cached_league_managers(TEAMS[team_a]["league_id"])
    mb = cached_league_managers(TEAMS[team_b]["league_id"])
    ids_a = tuple(str(m["id"]) for m in ma)
    ids_b = tuple(str(m["id"]) for m in mb)

    # CRITICAL: GW N analysis uses GW N-1 squad state.
    pa = cached_team_picks(ids_a, -1, max(1, cutoff), "A", player_map, ma)
    pb = cached_team_picks(ids_b, -1, max(1, cutoff), "B", player_map, mb)

    def ownership(picks):
        out = defaultdict(int)
        for mgr in picks:
            for p in mgr.get("picks", []):
                out[int(p["element"])] += 1
        return out

    oa, ob = ownership(pa), ownership(pb)
    player_ids = tuple(sorted(set(oa) | set(ob)))
    summaries = cached_player_summaries(player_ids)
    analytics_rows = {r["id"]: r for r in []}

    # Fixture index for GW N onward.
    fx_idx = _fixture_index(fixture_data)
    rows = []
    for pid in player_ids:
        p = next((x for x in bootstrap.get("elements", []) if int(x["id"]) == pid), None)
        if not p:
            continue
        pos = engine.POSITIONS.get(p.get("element_type"), "?")
        metrics = _pre_gw_player_metric(pid, (summaries.get(pid) or {}).get("history", []), cutoff)
        threshold = _defcon_threshold(pos)
        played = metrics["history"]
        defcon_hits = [h for h in played if _safe_float(h.get("minutes")) > 0 and _safe_float(h.get("defensive_contribution")) >= threshold]
        def_rel = len(defcon_hits) / max(1, len([h for h in played if _safe_float(h.get("minutes")) > 0])) * 100.0

        next_fixtures = [f for f in fx_idx.get(int(p.get("team") or 0), []) if f["event"] >= int(gw) and not f["finished"]][:max(1, horizon)]
        fixture = next_fixtures[0] if next_fixtures else None
        fdr = int(fixture["fdr"]) if fixture else 3
        # Same transparent style as the core projection, but strictly historical through N-1.
        baseline = metrics["recent_ppg"]
        underlying = 1.55 * metrics["xgi90"] + 0.10 * metrics["dc90"]
        minutes_factor = 0.62 + 0.38 * metrics["minutes_security"] / 100.0
        proj = max(0.0, (baseline + underlying) * minutes_factor * _difficulty_factor(fdr))
        if pos in ("GKP", "DEF"):
            proj += 0.18 * (def_rel / 100.0)

        ict = metrics["ict"]
        # ICT is normalized against the observed scout pool later; keep raw FPL ICT here.
        own_a, own_b = oa.get(pid, 0), ob.get(pid, 0)
        diff = own_a - own_b
        swing = diff * proj
        rows.append({
            "id": pid,
            "name": p.get("web_name") or f"#{pid}",
            "position": pos,
            "club": teams.get(int(p.get("team") or 0), "?"),
            "mine": own_a,
            "opp": own_b,
            "diff": diff,
            "projected": proj,
            "minutes_security": metrics["minutes_security"],
            "xg90": metrics["xg90"],
            "xa90": metrics["xa90"],
            "xgi90": metrics["xgi90"],
            "defcon_reliability": def_rel,
            "fdr": fdr,
            "ict": ict,
            "influence": metrics["influence"],
            "creativity": metrics["creativity"],
            "threat": metrics["threat"],
            "expected_swing": swing,
            "fixture": fixture,
        })

    # ICT percentile is relative to this actual matchup pool, not the whole universe.
    ict_sorted = sorted([r["ict"] for r in rows])
    for r in rows:
        if ict_sorted:
            le = sum(v <= r["ict"] for v in ict_sorted)
            r["ict_percentile"] = 100.0 * le / len(ict_sorted)
        else:
            r["ict_percentile"] = 0.0

    rows.sort(key=lambda r: (-abs(r["expected_swing"]), -r["projected"], r["name"]))
    return rows, ma, mb, pa, pb, cutoff


def render_league_analytics(gw: int, horizon: int, ownership_threshold: float):
    st.subheader("⚔️ Differential Board — Next Opponent Scout")
    st.caption(
        f"Pre-GW{gw} scouting: player statistics and squad state are locked through GW{max(0, gw-1)}. "
        "GW{0}+ is used only for fixtures and forward projection.".format(gw)
    )

    # Pick the user's IML side, then automatically identify the PL/IML opponent for the selected GW.
    team_names = sorted(TEAMS.keys())
    default_team = st.selectbox("My IML team", team_names, key="scout_my_team")
    matchups = gw_matchups(int(gw))
    auto_opp = None
    for m in matchups:
        if m["team_a"] == default_team:
            auto_opp = m["team_b"]
            break
        if m["team_b"] == default_team:
            auto_opp = m["team_a"]
            break
    candidates = [x for x in team_names if x != default_team]
    opponent = st.selectbox("Opponent", candidates, index=(candidates.index(auto_opp) if auto_opp in candidates else 0), key="scout_opponent")

    if st.button(f"Load GW{gw} matchup", type="primary", key="scout_load"):
        st.session_state["scout_loaded"] = (default_team, opponent, int(gw), int(horizon))

    loaded = st.session_state.get("scout_loaded")
    if not loaded:
        st.info("Select your team and opponent, then load the matchup.")
        return
    my_team, opp_team, load_gw, load_horizon = loaded
    try:
        with st.spinner("Loading the 8 squads and building the differential table…"):
            rows, managers_a, managers_b, picks_a, picks_b, cutoff = _build_simple_scout_rows(my_team, opp_team, load_gw, load_horizon)
    except Exception as e:
        st.error(f"Couldn't build the matchup: {e}")
        return

    if not rows:
        st.warning("No players could be loaded for this matchup.")
        return

    projected_a = sum(r["projected"] * r["mine"] for r in rows)
    projected_b = sum(r["projected"] * r["opp"] for r in rows)
    margin = projected_a - projected_b
    result_state = "WIN" if margin >= 6 else "DRAW" if margin >= -5 else "LOSS"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(my_team, f"{projected_a:.1f}")
    c2.metric(opp_team, f"{projected_b:.1f}")
    c3.metric("Projected margin", f"{margin:+.1f}")
    c4.metric("IML state", result_state)

    st.markdown(f"### {my_team} vs {opp_team} — GW{load_gw}")
    st.caption(f"Stats cutoff: GW{cutoff}. FDR/fixture shown for GW{load_gw}; no GW{load_gw} performance is used.")

    df = pd.DataFrame(rows)
    show = df[[
        "name", "position", "club", "mine", "opp", "diff", "projected",
        "minutes_security", "xg90", "xa90", "xgi90", "defcon_reliability",
        "ict", "ict_percentile", "fdr", "expected_swing"
    ]].copy()
    show.columns = [
        "Player", "Pos", "Club", "Mine", "Opp", "Diff", "xPts",
        "Min Security %", "xG90", "xA90", "xGI90", "DEFCON Rel. %",
        "ICT", "ICT %ile", "FDR", "Expected Swing"
    ]
    for col in ["xPts", "Min Security %", "xG90", "xA90", "xGI90", "DEFCON Rel. %", "ICT", "ICT %ile", "Expected Swing"]:
        show[col] = show[col].round(2)

    def _swing_style(v):
        if v > 0:
            return "color: #1a8754; font-weight: 700"
        if v < 0:
            return "color: #c0392b; font-weight: 700"
        return "color: #6b7280"

    st.dataframe(show.style.map(_swing_style, subset=["Expected Swing"]), hide_index=True, width="stretch", height=620)

    st.markdown("#### Swing Board")
    chart = df.head(15).sort_values("expected_swing")
    fig = go.Figure(go.Bar(
        x=chart["expected_swing"], y=chart["name"], orientation="h",
        text=[f"{x:+.1f}" for x in chart["expected_swing"]], textposition="outside"
    ))
    fig.add_vline(x=0, line_dash="dot")
    fig.update_layout(height=max(360, 26 * len(chart)), margin=dict(l=0, r=50, t=20, b=0), xaxis_title="Expected H2H swing (pts)")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Biggest matchup levers")
    biggest_for = sorted([r for r in rows if r["expected_swing"] > 0], key=lambda r: -r["expected_swing"])[:3]
    biggest_against = sorted([r for r in rows if r["expected_swing"] < 0], key=lambda r: r["expected_swing"])[:3]
    a, b = st.columns(2)
    with a:
        st.markdown("**🟢 Opportunities**")
        for r in biggest_for:
            st.write(f"**{r['name']}** · +{r['expected_swing']:.1f} swing · ICT {r['ict']:.0f} · xPts {r['projected']:.1f}")
    with b:
        st.markdown("**🔴 Threats**")
        for r in biggest_against:
            st.write(f"**{r['name']}** · {r['expected_swing']:.1f} swing · ICT {r['ict']:.0f} · xPts {r['projected']:.1f}")


# ── Main ───────────────────────────────────────────────────────────────────────
st.title("IML Scorecards")
st.caption("H2H ownership, player projections and IML-wide decision intelligence — 2026/27")

if mode == "League Analytics":
    render_league_analytics(int(gw), int(horizon), float(ownership_threshold))
else:
    render_weekly_matchups(int(gw), no_live, no_summary, no_chips)
