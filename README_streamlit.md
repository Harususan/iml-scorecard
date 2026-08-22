# FPL H2H Scout — Streamlit app

## Setup

```bash
pip install streamlit plotly pandas reportlab
```

(`fpl_h2h_v14.py` and `streamlit_app.py` must be in the same folder.)

## Run

```bash
streamlit run streamlit_app.py
```

This opens the app in your browser at `http://localhost:8501`.

## Using it

1. In the sidebar, enter:
   - **Gameweek**
   - **League A ID** and **League B ID** — the number in your league's URL
     (`fantasy.premierleague.com/leagues/<ID>/standings/c`)
   - **H2H captain** for each side — the manager's name (partial match works,
     e.g. "Alice" matches "Alice Smith")
2. Click **Run analysis**.
3. Browse the tabs: Scoreboard, Team Totals, Differential & Swing, Points Race,
   Chip Tracker, Squads.
4. Download a polished **PDF report** or the raw **JSON** data from the
   Scoreboard tab.

## Notes

- Data is cached briefly (live scores: 60s, league/history data: a few
  minutes, player data: 1 hour) so flipping between tabs or re-running with
  the same inputs doesn't hammer the FPL API.
- "Skip live scores", "Skip squad summaries", and "Skip chip tracker" in the
  sidebar mirror the `--no-live`, `--no-summary`, and chip-tracker options
  from the original CLI script.
- The original CLI script (`fpl_h2h_v14.py`) still works standalone from the
  command line — nothing about its `main()`/argparse path was changed, it's
  only imported as a library here.
