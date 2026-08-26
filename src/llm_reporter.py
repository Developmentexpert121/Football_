"""
Stage I: Local LLM Report Generation via Ollama.

Connects to a locally-running Ollama instance to generate AI-written
tactical analysis of the match using structured match statistics as input.

- Supports Qwen3 8B, Gemma 3 12B, Llama 3.1 8B, or any Ollama-hosted model
- Sends match stats JSON → LLM produces 2-3 paragraphs of tactical insight
- Gracefully falls back to template-based report if Ollama is not running
- No API key, no cloud dependency — fully offline

Install Ollama:
    https://ollama.com/download
    ollama pull qwen3:8b
"""

import json
from typing import Dict, Any, Optional

# Graceful fallback if Ollama Python client not installed
try:
    import ollama as ollama_client
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False

# Also try raw HTTP as fallback
try:
    import urllib.request
    import urllib.error
    URLLIB_AVAILABLE = True
except ImportError:
    URLLIB_AVAILABLE = False


# Default Ollama endpoint
OLLAMA_DEFAULT_URL = "http://localhost:11434"

# Model priority (tries in order — first available wins)
PREFERRED_MODELS = [
    'qwen3:8b',
    'gemma3:12b',
    'llama3.1:8b',
    'mistral:7b',
    'phi3:mini',
]

# System prompt for football analytics LLM
SYSTEM_PROMPT = """You are an expert football (soccer) tactical analyst with 25+ years of experience 
analyzing professional matches. You write insightful, concise match reports that combine statistical 
data with tactical understanding.

Your analysis should cover:
1. Overall match narrative (who dominated, key phases)
2. Tactical observations (formation effectiveness, pressing, defensive shape)
3. Standout individual performances (distance, speed, touches)
4. Key events and turning points

Write 3-4 paragraphs. Use specific numbers from the data. Be authoritative but accessible.
Do NOT use bullet points — write in flowing prose paragraphs.
Do NOT repeat raw data tables — weave the statistics naturally into your narrative."""


class LLMReporter:
    """
    Generates AI-written match analysis reports using a local Ollama LLM.

    Falls back gracefully to template-based report if:
    - Ollama is not installed
    - Ollama server is not running
    - No suitable model is available
    - LLM generation fails for any reason
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        ollama_url: str = OLLAMA_DEFAULT_URL,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name
        self._available = False
        self._method = None  # 'client' or 'http'

        # Try to connect to Ollama
        if OLLAMA_CLIENT_AVAILABLE:
            try:
                models = ollama_client.list()
                available_models = [m.get('name', '') for m in models.get('models', [])]
                self.model_name = self._select_model(available_models, model_name)
                if self.model_name:
                    self._available = True
                    self._method = 'client'
                    print(f"[LLMReporter] Ollama connected. Using model: {self.model_name}")
                else:
                    print(f"[LLMReporter] Ollama running but no suitable model found.")
                    print(f"             Available: {available_models}")
                    print(f"             Install one with: ollama pull qwen3:8b")
            except Exception as e:
                print(f"[LLMReporter] Ollama client connection failed: {e}")

        # Fallback: try raw HTTP
        if not self._available and URLLIB_AVAILABLE:
            try:
                req = urllib.request.Request(
                    f"{ollama_url}/api/tags",
                    method='GET'
                )
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = json.loads(resp.read().decode())
                    available_models = [m.get('name', '') for m in data.get('models', [])]
                    self.model_name = self._select_model(available_models, model_name)
                    if self.model_name:
                        self._available = True
                        self._method = 'http'
                        print(f"[LLMReporter] Ollama HTTP connected. Using model: {self.model_name}")
            except Exception:
                pass

        if not self._available:
            print("[LLMReporter] Ollama not available. Will use template-based reports.")
            print("             Install Ollama: https://ollama.com/download")
            print("             Then run: ollama pull qwen3:8b")

    def _select_model(self, available: list, preferred: Optional[str] = None) -> Optional[str]:
        """Selects the best available model from the preferred list."""
        if preferred and any(preferred in m for m in available):
            return preferred

        for model in PREFERRED_MODELS:
            for avail in available:
                if model in avail:
                    return avail
        
        # If any model is available, use the first one
        return available[0] if available else None

    def generate_report(
        self,
        analytics_data: Dict[str, Any],
        tactical_data: Optional[Dict[str, Any]] = None,
        events_data: Optional[list] = None,
        team_names: tuple = ('Team A', 'Team B')
    ) -> Dict[str, Any]:
        """
        Generates an AI-written match analysis report.

        Args:
            analytics_data: Output from AnalyticsEngine
            tactical_data: Output from TacticalAnalyzer (optional)
            events_data: List of detected events (optional)
            team_names: Tuple of team names for personalization

        Returns:
            Dict with keys:
                'report_text': str — the generated report
                'model_used': str — model name or 'template'
                'method': str — 'ollama' or 'template'
        """
        # Build structured context for LLM
        context = self._build_context(analytics_data, tactical_data, events_data, team_names)

        if self._available:
            try:
                report_text = self._generate_with_llm(context)
                if report_text and len(report_text) > 100:
                    return {
                        'report_text': report_text,
                        'model_used': self.model_name,
                        'method': 'ollama'
                    }
            except Exception as e:
                print(f"[LLMReporter] LLM generation failed: {e}. Falling back to template.")

        # Fallback: template-based report
        report_text = self._generate_template_report(analytics_data, tactical_data, events_data, team_names)
        return {
            'report_text': report_text,
            'model_used': 'template',
            'method': 'template'
        }

    def _build_context(
        self,
        analytics_data: Dict[str, Any],
        tactical_data: Optional[Dict[str, Any]],
        events_data: Optional[list],
        team_names: tuple
    ) -> str:
        """Builds a structured text context to send to the LLM."""
        parts = []
        parts.append(f"MATCH ANALYSIS DATA — {team_names[0]} vs {team_names[1]}")
        parts.append("=" * 50)

        # Possession
        poss = analytics_data.get('possession_stats', {})
        parts.append(f"\nPossession: {team_names[0]} {poss.get('team_a_possession_pct', 50)}% — "
                      f"{team_names[1]} {poss.get('team_b_possession_pct', 50)}%")

        # Player performance summary
        player_stats = analytics_data.get('player_stats', {})
        team_stats: Dict[int, Dict[str, list]] = {0: {'dist': [], 'speed': [], 'max_speed': []},
                                                    1: {'dist': [], 'speed': [], 'max_speed': []}}

        parts.append(f"\nTotal Players Tracked: {len(player_stats)}")
        for pid, stats in player_stats.items():
            tid = stats.get('team_id', 0)
            team_stats[tid]['dist'].append(stats.get('total_distance_m', 0))
            team_stats[tid]['speed'].append(stats.get('avg_speed_km_h', 0))
            team_stats[tid]['max_speed'].append(stats.get('max_speed_km_h', 0))
            sprint_count = stats.get('sprint_count', 0)
            parts.append(f"  Player #{pid} ({team_names[tid]}): "
                          f"Dist={stats.get('total_distance_m', 0)}m, "
                          f"AvgSpeed={stats.get('avg_speed_km_h', 0)}km/h, "
                          f"MaxSpeed={stats.get('max_speed_km_h', 0)}km/h"
                          f"{f', Sprints={sprint_count}' if sprint_count else ''}")

        for tid in [0, 1]:
            if team_stats[tid]['dist']:
                import numpy as np
                parts.append(f"\n{team_names[tid]} Team Summary:")
                parts.append(f"  Total Distance: {sum(team_stats[tid]['dist']):.0f}m")
                parts.append(f"  Avg Player Speed: {np.mean(team_stats[tid]['speed']):.1f} km/h")
                parts.append(f"  Fastest Sprint: {max(team_stats[tid]['max_speed']):.1f} km/h")

        # Tactical data
        if tactical_data:
            parts.append("\nTACTICAL DATA:")
            formations = tactical_data.get('formations', {})
            parts.append(f"  {team_names[0]} Formation: {formations.get('team_0', 'Unknown')}")
            parts.append(f"  {team_names[1]} Formation: {formations.get('team_1', 'Unknown')}")

            zones = tactical_data.get('attacking_zones', {})
            parts.append(f"  Ball in Defensive Third: {zones.get('defensive_third_pct', 0)}%")
            parts.append(f"  Ball in Midfield: {zones.get('midfield_third_pct', 0)}%")
            parts.append(f"  Ball in Attacking Third: {zones.get('attacking_third_pct', 0)}%")

            pressing = tactical_data.get('pressing_intensity', {})
            parts.append(f"  Pressing: avg {pressing.get('avg_players_near_ball', 0):.1f} players near ball")

            passes = tactical_data.get('passing_network', [])
            if passes:
                parts.append(f"  Top passing connection: Player #{passes[0]['from_player']} → "
                              f"#{passes[0]['to_player']} ({passes[0]['pass_count']} passes)")

        # Events
        if events_data:
            parts.append(f"\nKEY EVENTS ({len(events_data)} total):")
            for evt in events_data[:10]:
                parts.append(f"  [{evt.get('timestamp', '?')}] {evt.get('event_type', '?')}: "
                              f"{evt.get('description', '')}")

        return "\n".join(parts)

    def _generate_with_llm(self, context: str) -> str:
        """Generates report text using Ollama LLM."""
        prompt = f"""Based on the following match analysis data, write a professional tactical 
match report (3-4 paragraphs):

{context}

Write your analysis now:"""

        if self._method == 'client':
            response = ollama_client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                }
            )
            return response['message']['content']

        elif self._method == 'http':
            payload = json.dumps({
                'model': self.model_name,
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}
                ],
                'stream': False,
                'options': {
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                }
            }).encode()

            req = urllib.request.Request(
                f"{self.ollama_url}/api/chat",
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
                return data.get('message', {}).get('content', '')

        return ''

    def _generate_template_report(
        self,
        analytics_data: Dict[str, Any],
        tactical_data: Optional[Dict[str, Any]],
        events_data: Optional[list],
        team_names: tuple
    ) -> str:
        """Generates a template-based report when LLM is not available."""
        poss = analytics_data.get('possession_stats', {})
        player_stats = analytics_data.get('player_stats', {})

        team_a_poss = poss.get('team_a_possession_pct', 50.0)
        team_b_poss = poss.get('team_b_possession_pct', 50.0)

        dominant_team = team_names[0] if team_a_poss > team_b_poss else team_names[1]
        dominant_poss = max(team_a_poss, team_b_poss)

        # Find key performers
        max_dist_player = max(player_stats.items(), key=lambda x: x[1].get('total_distance_m', 0), default=(None, {}))
        max_speed_player = max(player_stats.items(), key=lambda x: x[1].get('max_speed_km_h', 0), default=(None, {}))

        report_lines = [
            f"MATCH REPORT: {team_names[0]} vs {team_names[1]}",
            "=" * 50,
            "",
            f"{dominant_team} dominated possession with {dominant_poss:.1f}% of the ball, "
            f"dictating the tempo throughout the analyzed passage of play. "
            f"The {team_names[0]} ({team_a_poss:.1f}%) versus {team_names[1]} ({team_b_poss:.1f}%) "
            f"possession split tells a clear story of territorial control.",
            "",
        ]

        if max_dist_player[0] is not None:
            p_id = max_dist_player[0]
            p_stats = max_dist_player[1]
            team_label = team_names[p_stats.get('team_id', 0)]
            report_lines.append(
                f"Player #{p_id} ({team_label}) covered the most ground at "
                f"{p_stats.get('total_distance_m', 0):.0f}m with an average speed of "
                f"{p_stats.get('avg_speed_km_h', 0):.1f} km/h — indicating high work rate "
                f"and defensive contribution."
            )

        if max_speed_player[0] is not None:
            p_id = max_speed_player[0]
            p_stats = max_speed_player[1]
            team_label = team_names[p_stats.get('team_id', 0)]
            report_lines.append(
                f"The fastest player was #{p_id} ({team_label}) with a top sprint speed of "
                f"{p_stats.get('max_speed_km_h', 0):.1f} km/h, showing explosive pace in transitions."
            )

        if tactical_data:
            formations = tactical_data.get('formations', {})
            report_lines.append("")
            report_lines.append(
                f"Tactically, {team_names[0]} set up in a {formations.get('team_0', 'fluid')} "
                f"formation while {team_names[1]} opted for a {formations.get('team_1', 'compact')} shape. "
            )

            zones = tactical_data.get('attacking_zones', {})
            atk_pct = zones.get('attacking_third_pct', 0)
            if atk_pct > 40:
                report_lines.append(
                    f"The attacking third saw significant action ({atk_pct}% of ball time), "
                    f"suggesting an aggressive, front-foot approach."
                )

        if events_data:
            n_events = len(events_data)
            event_types = set(e.get('event_type', '') for e in events_data)
            report_lines.append("")
            report_lines.append(
                f"The analysis flagged {n_events} notable events including "
                f"{', '.join(event_types)}. These incidents shaped the flow of play "
                f"and required referee intervention."
            )

        report_lines.append("")
        report_lines.append("(Report generated by Football Analytics AI Engine)")

        return "\n".join(report_lines)
