"""
Stage 10 (Part B): Enhanced Report Generation.

Exports comprehensive analytics results to CSV datasets and builds a self-contained,
interactive HTML match report dashboard with:
- Enhanced player stats (sprints, acceleration, touches, passes)
- All event types (goals, corners, offside, shots, penalty area entries)
- Tactical analysis data (formations, pressing, attacking zones, passing network)
- Per-team heatmap links
"""

import os
import csv
import json
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from typing import Dict, Any, List, Optional


class ReportGenerator:
    """
    Exports analytics results to CSV datasets and builds interactive HTML dashboard.
    """
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_csv(
        self,
        analytics_data: Dict[str, Any],
        events_data: List[Dict[str, Any]],
        tactical_data: Optional[Dict[str, Any]] = None,
        summary_stats: Optional[Dict[str, Any]] = None
    ):
        """
        Exports player statistics, event detections, and tactical data to CSV/JSON files.
        """
        player_stats = analytics_data.get('player_stats', {})
        
        # 1. Player Stats CSV (Enhanced with new metrics)
        rows = []
        for p_id, stats in player_stats.items():
            rows.append({
                'Player_ID': p_id,
                'Jersey_Number': stats.get('jersey_number', stats.get('jersey', p_id)),
                'Team_ID': stats['team_id'],
                'Team_Name': "Team A (Red)" if stats['team_id'] == 0 else "Team B (Blue)",
                'Total_Distance_Meters': stats['total_distance_m'],
                'Avg_Speed_km_h': stats['avg_speed_km_h'],
                'Max_Speed_km_h': stats['max_speed_km_h'],
                'Sprint_Count': stats.get('sprint_count', 0),
                'Sprint_Time_Sec': stats.get('sprint_time_sec', 0),
                'Avg_Acceleration_ms2': stats.get('avg_acceleration_ms2', 0),
                'Max_Acceleration_ms2': stats.get('max_acceleration_ms2', 0),
                'Max_Deceleration_ms2': stats.get('max_deceleration_ms2', 0),
                'High_Intensity_Changes': stats.get('high_intensity_changes', 0),
                'Touch_Count': stats.get('touch_count', 0),
                'Pass_Count': stats.get('pass_count', 0),
            })
        
        csv_players_path = os.path.join(self.output_dir, "stats_player.csv")
        if PANDAS_AVAILABLE:
            df_players = pd.DataFrame(rows)
            df_players.to_csv(csv_players_path, index=False)
        else:
            if rows:
                with open(csv_players_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

        print(f"Player stats CSV saved to: {csv_players_path}")

        # 2. Events CSV (Enhanced with all event types)
        csv_events_path = os.path.join(self.output_dir, "events.csv")
        event_rows = []
        for event in events_data:
            event_rows.append({
                'Frame': event.get('frame_idx', ''),
                'Timestamp': event.get('timestamp', ''),
                'Timestamp_Seconds': event.get('timestamp_seconds', ''),
                'Event_Type': event.get('event_type', ''),
                'Players_Involved': str(event.get('players_involved', [])),
                'Teams_Involved': str(event.get('teams_involved', [])),
                'Confidence': event.get('confidence', 0),
                'Description': event.get('description', '')
            })

        if PANDAS_AVAILABLE:
            df_events = pd.DataFrame(event_rows)
            df_events.to_csv(csv_events_path, index=False)
        else:
            if event_rows:
                with open(csv_events_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=event_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(event_rows)

        print(f"Match events CSV saved to: {csv_events_path}")

        # 3. Tactical Data JSON
        if tactical_data:
            tactical_json_path = os.path.join(self.output_dir, "tactical_data.json")
            # Convert non-serializable types
            clean_tactical = json.loads(json.dumps(tactical_data, default=str))
            with open(tactical_json_path, "w", encoding="utf-8") as f:
                json.dump(clean_tactical, f, indent=2, ensure_ascii=False)
            print(f"Tactical data JSON saved to: {tactical_json_path}")

        # 4. Summary Stats JSON
        if summary_stats:
            summary_json_path = os.path.join(self.output_dir, "summary_stats.json")
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(summary_stats, f, indent=2, ensure_ascii=False)
            print(f"Summary stats JSON saved to: {summary_json_path}")

    def generate_html_dashboard(
        self,
        analytics_data: Dict[str, Any],
        events_data: List[Dict[str, Any]],
        tactical_data: Optional[Dict[str, Any]] = None,
        output_filename: str = "dashboard.html"
    ):
        """
        Generates comprehensive, self-contained HTML match report with all analytics.
        """
        player_stats = analytics_data.get('player_stats', {})
        possession = analytics_data.get('possession_stats', {})

        team_a_possession = possession.get('team_a_possession_pct', 50.0)
        team_b_possession = possession.get('team_b_possession_pct', 50.0)

        # Build player stats table rows
        player_table_html = ""
        for p_id, stats in sorted(player_stats.items(), key=lambda x: x[1].get('total_distance_m', 0), reverse=True):
            team_badge = f'<span class="badge team-a">Team A</span>' if stats['team_id'] == 0 else f'<span class="badge team-b">Team B</span>'
            sprint_badge = f'<span class="sprint-badge">{stats.get("sprint_count", 0)} sprints</span>' if stats.get('sprint_count', 0) > 0 else ''
            player_table_html += f"""
            <tr>
                <td>Player #{p_id}</td>
                <td>{team_badge}</td>
                <td>{stats['total_distance_m']} m</td>
                <td>{stats['avg_speed_km_h']} km/h</td>
                <td><strong>{stats['max_speed_km_h']} km/h</strong></td>
                <td>{stats.get('sprint_count', 0)} {sprint_badge}</td>
                <td>{stats.get('touch_count', 0)}</td>
                <td>{stats.get('pass_count', 0)}</td>
                <td>{stats.get('avg_acceleration_ms2', 0)} m/s²</td>
            </tr>
            """

        # Build event cards
        event_type_badges = {
            'Potential Foul': ('foul-badge', '⚠️'),
            'Yellow Card Candidate': ('card-badge', '🟡'),
            'Goal': ('goal-badge', '⚽'),
            'Shot on Target': ('shot-badge', '🎯'),
            'Corner Kick': ('corner-badge', '🚩'),
            'Free Kick': ('freekick-badge', '🔵'),
            'Offside': ('offside-badge', '🚫'),
            'Penalty Area Entry': ('penalty-badge', '📦'),
        }

        event_cards_html = ""
        if not events_data:
            event_cards_html = "<p class='text-muted'>No major incidents flagged during this match clip.</p>"
        else:
            for event in events_data:
                badge_class, emoji = event_type_badges.get(event['event_type'], ('foul-badge', '📋'))
                event_cards_html += f"""
                <div class="event-card">
                    <div class="event-header">
                        <span class="badge {badge_class}">{emoji} {event['event_type']}</span>
                        <span class="timestamp">⏱ {event['timestamp']}</span>
                    </div>
                    <p class="event-desc">{event['description']}</p>
                    <small>Confidence Score: <strong>{int(event['confidence']*100)}%</strong></small>
                </div>
                """

        # Tactical section
        tactical_html = ""
        if tactical_data:
            formations = tactical_data.get('formations', {})
            zones = tactical_data.get('attacking_zones', {})
            pressing = tactical_data.get('pressing_intensity', {})
            compactness = tactical_data.get('compactness', {})
            width_depth = tactical_data.get('width_depth', {})
            wd_0 = (width_depth or {}).get('team_0', {}) or {}
            wd_1 = (width_depth or {}).get('team_1', {}) or {}

            # Passing network
            passing_html = ""
            for p in tactical_data.get('passing_network', [])[:8]:
                team_label = "A" if p['team'] == 0 else "B"
                passing_html += f"<tr><td>#{p['from_player']}</td><td>#{p['to_player']}</td><td>{p['pass_count']}</td><td>Team {team_label}</td></tr>"

            tactical_html = f"""
            <div class="grid">
                <div class="card">
                    <h3>⚔️ Formations</h3>
                    <div class="formation-display">
                        <div class="formation-team">
                            <span class="badge team-a">Team A</span>
                            <span class="formation-value">{formations.get('team_0', 'Unknown')}</span>
                        </div>
                        <div class="formation-team">
                            <span class="badge team-b">Team B</span>
                            <span class="formation-value">{formations.get('team_1', 'Unknown')}</span>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <h3>📊 Attacking Zones</h3>
                    <div class="zone-bar">
                        <div class="zone def-zone" style="width:{zones.get('defensive_third_pct', 33)}%">{zones.get('defensive_third_pct', 33)}%</div>
                        <div class="zone mid-zone" style="width:{zones.get('midfield_third_pct', 34)}%">{zones.get('midfield_third_pct', 34)}%</div>
                        <div class="zone atk-zone" style="width:{zones.get('attacking_third_pct', 33)}%">{zones.get('attacking_third_pct', 33)}%</div>
                    </div>
                    <div class="zone-labels">
                        <span>Defensive</span><span>Midfield</span><span>Attacking</span>
                    </div>
                </div>
            </div>
            <div class="grid">
                <div class="card">
                    <h3>🔥 Pressing & Shape</h3>
                    <p>Avg players near ball: <strong>{pressing.get('avg_players_near_ball', 0):.1f}</strong></p>
                    <p>Max pressing intensity: <strong>{pressing.get('max_players_near_ball', 0)}</strong> players</p>
                    <hr style="border-color:#334155">
                    <p>Team A compactness: <strong>{compactness.get('team_0', 0)} m²</strong></p>
                    <p>Team B compactness: <strong>{compactness.get('team_1', 0)} m²</strong></p>
                    <p>Team A width/depth: {wd_0.get('avg_width_m', 0)}m / {wd_0.get('avg_depth_m', 0)}m</p>
                    <p>Team B width/depth: {wd_1.get('avg_width_m', 0)}m / {wd_1.get('avg_depth_m', 0)}m</p>
                </div>
                <div class="card">
                    <h3>🔗 Passing Network (Top Connections)</h3>
                    <table>
                        <thead><tr><th>From</th><th>To</th><th>Passes</th><th>Team</th></tr></thead>
                        <tbody>{passing_html}</tbody>
                    </table>
                </div>
            </div>
            """

        # Event summary counts
        event_summary = {}
        for e in events_data:
            et = e['event_type']
            event_summary[et] = event_summary.get(et, 0) + 1

        event_summary_html = ""
        for et, count in sorted(event_summary.items(), key=lambda x: -x[1]):
            badge_class, emoji = event_type_badges.get(et, ('foul-badge', '📋'))
            event_summary_html += f'<div class="stat-pill"><span class="badge {badge_class}">{emoji} {et}</span><span class="stat-count">{count}</span></div>'

        dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Football Match Analytics Dashboard — 18-Stage AI Analysis</title>
    <meta name="description" content="Comprehensive football match analytics dashboard powered by 18-stage AI pipeline with pose estimation, tactical analysis, and event detection.">
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --bg-card-hover: #263548;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --accent-green: #22c55e;
            --accent-yellow: #eab308;
            --accent-purple: #a855f7;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
            line-height: 1.6;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .header h1 {{ margin: 0; font-size: 26px; }}
        .header p {{ margin: 4px 0 0 0; color: var(--text-muted); }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }}
        .card {{
            background-color: var(--bg-card);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            transition: background-color 0.2s;
        }}
        .card:hover {{ background-color: var(--bg-card-hover); }}
        .card h3 {{ margin-top: 0; font-size: 18px; color: #cbd5e1; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
        .possession-bar {{
            display: flex;
            height: 32px;
            border-radius: 8px;
            overflow: hidden;
            margin: 16px 0;
            font-weight: bold;
            font-size: 13px;
        }}
        .team-a-bar {{ background: linear-gradient(135deg, var(--accent-red), #dc2626); width: {team_a_possession}%; display: flex; align-items: center; justify-content: center; }}
        .team-b-bar {{ background: linear-gradient(135deg, var(--accent-blue), #2563eb); width: {team_b_possession}%; display: flex; align-items: center; justify-content: center; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 14px;
        }}
        th, td {{
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }}
        th {{ color: var(--text-muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .badge {{
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
            display: inline-block;
        }}
        .team-a {{ background-color: rgba(239, 68, 68, 0.2); color: #fca5a5; }}
        .team-b {{ background-color: rgba(59, 130, 246, 0.2); color: #93c5fd; }}
        .foul-badge {{ background-color: rgba(245, 158, 11, 0.2); color: #fde047; }}
        .card-badge {{ background-color: rgba(239, 68, 68, 0.3); color: #ef4444; }}
        .goal-badge {{ background-color: rgba(34, 197, 94, 0.3); color: #86efac; }}
        .shot-badge {{ background-color: rgba(168, 85, 247, 0.2); color: #d8b4fe; }}
        .corner-badge {{ background-color: rgba(59, 130, 246, 0.2); color: #93c5fd; }}
        .freekick-badge {{ background-color: rgba(6, 182, 212, 0.2); color: #67e8f9; }}
        .offside-badge {{ background-color: rgba(239, 68, 68, 0.2); color: #fca5a5; }}
        .penalty-badge {{ background-color: rgba(245, 158, 11, 0.2); color: #fcd34d; }}
        .sprint-badge {{ background-color: rgba(34, 197, 94, 0.15); color: #86efac; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-left: 4px; }}
        .event-card {{
            background: #0f172a;
            border-left: 4px solid #f59e0b;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 10px;
        }}
        .event-card:nth-child(odd) {{ border-left-color: var(--accent-purple); }}
        .event-header {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
        .timestamp {{ color: var(--text-muted); font-size: 13px; }}
        .event-desc {{ margin: 8px 0 4px 0; font-size: 14px; }}
        .stat-pill {{ display: inline-flex; align-items: center; gap: 8px; margin: 4px 8px 4px 0; padding: 4px 12px; background: rgba(255,255,255,0.05); border-radius: 20px; }}
        .stat-count {{ font-weight: bold; font-size: 16px; color: var(--accent-green); }}
        .formation-display {{ display: flex; gap: 20px; margin-top: 12px; }}
        .formation-team {{ display: flex; flex-direction: column; align-items: center; gap: 8px; flex: 1; }}
        .formation-value {{ font-size: 28px; font-weight: bold; color: var(--text-main); }}
        .zone-bar {{ display: flex; height: 28px; border-radius: 8px; overflow: hidden; margin: 12px 0 4px 0; font-size: 12px; font-weight: bold; }}
        .zone {{ display: flex; align-items: center; justify-content: center; }}
        .def-zone {{ background: #ef4444; }}
        .mid-zone {{ background: #eab308; color: #000; }}
        .atk-zone {{ background: #22c55e; color: #000; }}
        .zone-labels {{ display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); }}
        .section-title {{ font-size: 20px; font-weight: 700; margin: 32px 0 16px 0; padding-bottom: 8px; border-bottom: 2px solid var(--accent-purple); display: inline-block; }}
        hr {{ border: none; border-top: 1px solid var(--border); margin: 16px 0; }}
        @media (max-width: 768px) {{
            .grid {{ grid-template-columns: 1fr; }}
            body {{ padding: 12px; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>⚽ Football Match Video Analytics Dashboard</h1>
            <p>18-Stage AI Pipeline — Player Tracking, Pose Estimation, Tactical Analysis & Event Detection</p>
        </div>
    </div>

    <!-- Possession & Event Summary -->
    <div class="grid">
        <div class="card">
            <h3>🏟️ Ball Possession Breakdown</h3>
            <div class="possession-bar">
                <div class="team-a-bar">Team A ({team_a_possession}%)</div>
                <div class="team-b-bar">Team B ({team_b_possession}%)</div>
            </div>
            <p style="color: var(--text-muted); font-size: 14px; text-align: center;">Computed via spatial proximity mapping across broadcast frames.</p>
        </div>
        <div class="card">
            <h3>📊 Event Summary</h3>
            <div>{event_summary_html}</div>
        </div>
    </div>

    <!-- Tactical Analysis -->
    <h2 class="section-title">🧠 Tactical Analysis</h2>
    {tactical_html}

    <!-- Event Timeline -->
    <h2 class="section-title">📋 Match Event Timeline</h2>
    <div class="card">
        <h3>All Detected Events</h3>
        <div style="max-height: 400px; overflow-y: auto;">
            {event_cards_html}
        </div>
    </div>

    <!-- Player Performance -->
    <h2 class="section-title">👟 Player Performance Metrics</h2>
    <div class="card" style="overflow-x: auto;">
        <h3>Detailed Player Statistics</h3>
        <table>
            <thead>
                <tr>
                    <th>Player</th>
                    <th>Team</th>
                    <th>Distance</th>
                    <th>Avg Speed</th>
                    <th>Top Speed</th>
                    <th>Sprints</th>
                    <th>Touches</th>
                    <th>Passes</th>
                    <th>Acceleration</th>
                </tr>
            </thead>
            <tbody>
                {player_table_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        dashboard_path = os.path.join(self.output_dir, output_filename)
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(dashboard_html)
        print(f"Interactive HTML dashboard generated at: {dashboard_path}")
