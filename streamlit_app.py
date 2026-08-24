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


def find_next_gw(bootstrap: dict) -> int:
    """First unplayed gameweek, per bootstrap's own 'is_next' flag."""
    for ev in bootstrap.get("events", []):
        if ev.get("is_next"):
            return ev["id"]
    for ev in bootstrap.get("events", []):
        if not ev.get("finished"):
            return ev["id"]
    return 1


def find_next_opponent(team_name: str, gw: int):
    """Returns (opponent_team_name, is_home, fixture_dict) for `team_name`'s
    real PL fixture in gw, or (None, None, None) if not found (blank GW etc.)."""
    my_id = TEAM_TO_CLUB_ID.get(team_name)
    if my_id is None:
        return None, None, None
    for f in cached_raw_fixtures(gw):
        if f.get("team_h") == my_id:
            return CLUB_ID_TO_TEAM.get(f.get("team_a")), True, f
        if f.get("team_a") == my_id:
            return CLUB_ID_TO_TEAM.get(f.get("team_h")), False, f
    return None, None, None


@st.cache_data(ttl=600, show_spinner=False)
def cached_ep_map(_bootstrap):
    """element_id -> FPL's own 'expected points next fixture' (ep_next)."""
    ep = {}
    for el in _bootstrap.get("elements", []):
        try:
            ep[el["id"]] = float(el.get("ep_next") or 0)
        except (TypeError, ValueError):
            ep[el["id"]] = 0.0
    return ep


@st.cache_data(ttl=300, show_spinner=False)
def cached_history_squad_picks(manager_ids, gw, team_label, _player_map, managers):
    """Same as cached_team_picks but with cap_index=-1 (no H2H captain) — used
    in planner mode where the H2H captain for a future GW hasn't been chosen
    yet, so nobody's ownership count should be artificially doubled."""
    return engine.fetch_team_picks(manager_ids, -1, gw, team_label, _player_map, managers)


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


# ── Next-GW Planner mode: squad differential preview + captain/chip sim ──────
# def _projected_score(mgr: dict, player_map: dict, ep_map: dict, cap_name: str, chip: str) -> float:
#     starters = [p for p in mgr["picks"] if p["position"] <= 11]
#     bench    = [p for p in mgr["picks"] if p["position"] > 11]
#     pool = starters + bench if chip == "Bench Boost" else starters
#     total = 0.0
#     for p in pool:
#         pl   = player_map.get(p["element"], {})
#         xp   = ep_map.get(p["element"], 0.0)
#         mult = 1
#         if pl.get("name") == cap_name:
#             mult = 3 if chip == "Triple Captain" else 2
#         total += xp * mult
#     return round(total, 1)


# def render_planner_mode() -> None:
#     st.subheader("🔭 Next-GW Planner")
#     st.caption(
#         "Preview your squad against your next opponent before the deadline, and try "
#         "different captain/chip choices to shape your plan."
#     )

#     your_team = st.selectbox("Your team", TEAM_NAMES, key="planner_team")

#     bootstrap  = cached_bootstrap()
#     player_map = cached_player_map(bootstrap)
#     ep_map     = cached_ep_map(bootstrap)
#     next_gw    = find_next_gw(bootstrap)

#     try:
#         opp_team, is_home, fixture = find_next_opponent(your_team, next_gw)
#     except Exception as e:
#         st.error(f"Couldn't load GW{next_gw} fixtures: {e}")
#         return

#     if not opp_team:
#         st.warning(f"Couldn't find a GW{next_gw} fixture for {your_team} yet — could be a blank gameweek.")
#         return

#     venue = "🏠 Home" if is_home else "✈️ Away"
#     ko = fixture.get("kickoff_time") or ""
#     ko_display = f"{ko[:10]} {ko[11:16]} UTC" if len(ko) >= 16 else "Kickoff TBC"
#     st.info(f"**GW{next_gw}: {your_team} vs {opp_team}**  ·  {venue}  ·  {ko_display}")

#     try:
#         with st.spinner(f"Loading GW{next_gw} squads..."):
#             managers_you = cached_league_managers(TEAMS[your_team]["league_id"])
#             managers_opp = cached_league_managers(TEAMS[opp_team]["league_id"])
#             ids_you = [m["id"] for m in managers_you]
#             ids_opp = [m["id"] for m in managers_opp]
#             picks_you = cached_history_squad_picks(ids_you, next_gw, "A", player_map, managers_you)
#             picks_opp = cached_history_squad_picks(ids_opp, next_gw, "B", player_map, managers_opp)
#     except Exception as e:
#         st.warning(
#             f"Squads for GW{next_gw} aren't available yet ({e}). "
#             "They usually appear once the previous gameweek locks in."
#         )
#         return

#     rows = engine.build_differential(picks_you, picks_opp, player_map, {})
#     for r in rows:
#         r["ep_next"]    = ep_map.get(r["id"], 0.0)
#         r["proj_swing"] = round(r["diff"] * r["ep_next"], 1)
#     rows.sort(key=lambda r: -abs(r["proj_swing"]))

#     st.markdown("##### Squad differential preview")
#     if rows:
#         df = pd.DataFrame(rows)[["name", "position", "club", "A", "B", "diff", "ep_next", "proj_swing"]]
#         df.columns = ["Player", "Pos", "Club", "You", "Opponent", "Diff", "xPts (next)", "Proj. swing"]

#         def _swing_style(v):
#             if v > 0:
#                 return "color: #1a8754; font-weight: 600"
#             if v < 0:
#                 return "color: #c0392b; font-weight: 600"
#             return "color: #6b7280"

#         st.dataframe(
#             df.style.map(_swing_style, subset=["Proj. swing"]),
#             hide_index=True, width="stretch", height=380,
#         )
#     st.caption(
#         "xPts is FPL's own \u2018expected points, next fixture\u2019 model. Proj. swing = diff \u00d7 xPts — "
#         "a planning estimate, not a live score (this match hasn't been played)."
#     )

#     st.divider()
#     st.markdown("##### 🎯 Captain & chip simulator")
#     st.caption("Pick one of your own managers and try a different captain or chip to see the projected impact.")

#     your_managers = TEAMS[your_team]["managers"]
#     sim_name = st.selectbox("Simulate as", your_managers, key="planner_manager")
#     sim_idx  = your_managers.index(sim_name)
#     sim_mgr  = picks_you[sim_idx]

#     starters = [p for p in sim_mgr["picks"] if p["position"] <= 11]
#     starter_names = [player_map.get(p["element"], {}).get("name", f'#{p["element"]}') for p in starters]
#     current_cap_id   = next((p["element"] for p in sim_mgr["picks"] if p["is_captain"]), None)
#     current_cap_name = player_map.get(current_cap_id, {}).get("name", starter_names[0] if starter_names else "—")

#     c1, c2 = st.columns(2)
#     default_idx = starter_names.index(current_cap_name) if current_cap_name in starter_names else 0
#     cap_choice  = c1.selectbox("Captain", starter_names, index=default_idx, key="planner_cap")
#     chip_choice = c2.selectbox("Chip", ["None", "Triple Captain", "Bench Boost"], key="planner_chip")

#     current_projected = _projected_score(sim_mgr, player_map, ep_map, current_cap_name, "None")
#     sim_projected      = _projected_score(sim_mgr, player_map, ep_map, cap_choice, chip_choice)
#     delta = round(sim_projected - current_projected, 1)

#     m1, m2 = st.columns(2)
#     m1.metric("Current plan", f"{current_projected} xPts", help=f"Captain: {current_cap_name}, no chip")
#     m2.metric("Your simulation", f"{sim_projected} xPts", delta=f"{delta:+.1f}")

#     other_total_current = sum(
#         _projected_score(
#             mgr, player_map, ep_map,
#             player_map.get(next((p["element"] for p in mgr["picks"] if p["is_captain"]), None), {}).get("name", ""),
#             "None",
#         )
#         for i, mgr in enumerate(picks_you) if i != sim_idx
#     )
#     your_team_current = other_total_current + current_projected
#     your_team_sim      = other_total_current + sim_projected

#     opp_total = sum(
#         _projected_score(
#             mgr, player_map, ep_map,
#             player_map.get(next((p["element"] for p in mgr["picks"] if p["is_captain"]), None), {}).get("name", ""),
#             "None",
#         )
#         for mgr in picks_opp
#     )

#     st.write("")
#     st.markdown("###### Projected team totals (all 4 managers, no H2H captain assigned yet)")
#     t1, t2, t3 = st.columns(3)
#     t1.metric(f"{your_team} (current plan)", f"{your_team_current:.1f} xPts")
#     t2.metric(f"{your_team} (with your simulation)", f"{your_team_sim:.1f} xPts",
#               delta=f"{(your_team_sim - your_team_current):+.1f}")
#     t3.metric(f"{opp_team}", f"{opp_total:.1f} xPts")

#     margin = your_team_sim - opp_total
#     if margin > 0:
#         st.success(f"Projected: **{your_team}** ahead of **{opp_team}** by **{margin:.1f} xPts** with this plan.")
#     elif margin < 0:
#         st.error(f"Projected: **{opp_team}** ahead of **{your_team}** by **{abs(margin):.1f} xPts** with this plan.")
#     else:
#         st.info("Projected: dead level with this plan.")

#     st.caption(
#         "This is a planning estimate only — Wildcard/Free Hit aren't simulated here since they "
#         "change your whole squad, not just a scoring multiplier."
#     )


# ── Sidebar: global settings ───────────────────────────────────────────────────

# Streamlit only keeps a widget's session_state value alive while that widget
# is actually instantiated on every run. Since the Gameweek/skip-* controls
# only render in "This Week's Matchups" mode, we mirror them into plain
# (non-widget) session_state keys that survive switching to Planner mode
# and back.
for _k, _default in (("gw_value", 1), ("skip_live_value", False),
                      ("skip_summary_value", False), ("skip_chips_value", False)):
    if _k not in st.session_state:
        st.session_state[_k] = _default

with st.sidebar:
    st.header("⚽ IML Scorecards")
    mode = st.radio("Mode", ["This Week's Matchups"], key="app_mode")

    st.divider()
    if mode == "This Week's Matchups":
        st.header("Gameweek")
        gw = st.number_input("Gameweek", min_value=1, max_value=38,
                              value=st.session_state.gw_value, step=1, key="gw_input")
        st.session_state.gw_value = gw

        st.divider()
        no_live = st.checkbox("Skip live scores", value=st.session_state.skip_live_value, key="skip_live",
                               help="Use for a season that's fully finished, or to speed up testing.")
        st.session_state.skip_live_value = no_live
        no_summary = st.checkbox("Skip squad summaries", value=st.session_state.skip_summary_value, key="skip_summary")
        st.session_state.skip_summary_value = no_summary
        no_chips = st.checkbox("Skip chip tracker", value=st.session_state.skip_chips_value, key="skip_chips")
        st.session_state.skip_chips_value = no_chips

        st.caption(
            "Each PL fixture this gameweek gets its own tab, matched to the "
            "20-team H2H roster. Open a tab, pick both captains, then Analyse."
        )
    else:
        st.caption(
            "Pick your team to see your next PL fixture, your squad differential "
            "against that opponent, and simulate captain/chip choices ahead of the deadline."
        )


# ── Main ───────────────────────────────────────────────────────────────────────

st.title("IML Scorecards")
st.caption("Head-to-head ownership & point-swing analyser — 2026/27 season")

if mode == "This Week's Matchups":
    render_weekly_matchups(int(gw), no_live, no_summary, no_chips)
else:
    render_planner_mode()
