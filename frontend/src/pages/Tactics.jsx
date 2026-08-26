import { useParams } from 'react-router-dom'
import { useEffect, useState } from 'react'
import {
  Map, Network, TrendingUp, Users, ChevronDown, ChevronUp,
  Play, Pause, Target, Shield, Zap, Sparkles, Filter, Info,
  AlertCircle, CheckSquare, Square
} from 'lucide-react'

function getStoredMatch(jobId) {
  try {
    return JSON.parse(localStorage.getItem('fa_matches') || '[]').find(m => m.jobId === jobId) || null
  } catch { return null }
}

const TEAM_A_AVATAR = "/avatars/player_jersey_1.jpg"
const TEAM_B_AVATAR = "/avatars/player_jersey_4.jpg"

// Formation coordinates for 4-3-3 Passing Network (x, y percentages on pitch)
const FORMATION_433 = [
  { id: 1, name: "G. Donnarumma", role: "GK", x: 10, y: 50 },
  { id: 2, name: "A. Hakimi", role: "RB", x: 28, y: 18 },
  { id: 4, name: "Marquinhos", role: "CB", x: 25, y: 38 },
  { id: 5, name: "P. Kimpembe", role: "CB", x: 25, y: 62 },
  { id: 3, name: "N. Mendes", role: "LB", x: 28, y: 82 },
  { id: 6, name: "M. Verratti", role: "DM", x: 45, y: 50 },
  { id: 8, name: "Vitinha", role: "CM", x: 55, y: 30 },
  { id: 7, name: "Gareth G. Best", role: "CAM", x: 62, y: 70 }, // Active highlight player
  { id: 11, name: "O. Dembélé", role: "RW", x: 78, y: 20 },
  { id: 9, name: "K. Mbappé", role: "ST", x: 82, y: 50 },
  { id: 10, name: "B. Barcola", role: "LW", x: 78, y: 80 },
]

export default function Tactics() {
  const { jobId } = useParams()
  const [data, setData] = useState(null)
  const [heatmapTeam, setHeatmapTeam] = useState('a')
  const [timeWindow, setTimeWindow] = useState('Minutes 0-15')
  const [selectedPlayerId, setSelectedPlayerId] = useState(7)
  const [isPlayingScrubber, setIsPlayingScrubber] = useState(false)
  const [scrubberMin, setScrubberMin] = useState(25)
  const [aiInsightOpen, setAiInsightOpen] = useState(true)
  const [timeDropdownOpen, setTimeDropdownOpen] = useState(false)
  const [playerDropdownOpen, setPlayerDropdownOpen] = useState(false)
  const [passFilter, setPassFilter] = useState({ short: true, medium: false, long: false })

  useEffect(() => {
    const stored = getStoredMatch(jobId)
    if (stored) { setData(stored); return }
    fetch(`/api/results/${jobId}`)
      .then(r => r.json())
      .then(setData)
      .catch(() => {})
  }, [jobId])

  const selectedPlayer = FORMATION_433.find(p => p.id === selectedPlayerId) || FORMATION_433[7]
  const heatmapUrl = heatmapTeam === 'a' ? data?.heatmap_team_a_url : data?.heatmap_team_b_url

  return (
    <div className="min-h-screen tactics-pitch-bg transition-colors duration-300 -mt-6 pt-6 pb-12">
      <div className="max-w-7xl mx-auto px-4">
        {/* Page Title Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-black uppercase tracking-tight gradient-text">
            Tactics & Heatmap Analysis
          </h1>
          <p className="text-slate-500 dark:text-slate-400 text-xs mt-0.5 font-medium">
            Position analysis, passing network and match momentum
          </p>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
          {/* ========================================================
              LEFT COLUMN (6 cols): Main Heatmap + Goal Zone Heatmaps
              ======================================================== */}
          <div className="xl:col-span-6 flex flex-col gap-6">
            {/* Main Positional Heatmap Card */}
            <div className="glass tactics-card-glass p-5 rounded-2xl border border-white/10 shadow-xl relative flex flex-col">
              {/* Header Controls */}
              <div className="flex items-center justify-between flex-wrap gap-2 mb-4">
                <div className="flex items-center gap-2">
                  <Map size={15} className="text-[#00d4ff]" />
                  <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                    Position Positional Heatmap
                  </h2>
                </div>

                <div className="flex items-center gap-2">
                  {/* Team Switch Pills */}
                  <div className="flex bg-black/20 p-1 rounded-xl border border-white/10">
                    <button
                      onClick={() => setHeatmapTeam('a')}
                      className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                        heatmapTeam === 'a'
                          ? 'bg-[#0070f3] text-white shadow-[0_0_12px_rgba(0,112,243,0.5)]'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      Team A
                    </button>
                    <button
                      onClick={() => setHeatmapTeam('b')}
                      className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                        heatmapTeam === 'b'
                          ? 'bg-[#ef4444] text-white shadow-[0_0_12px_rgba(239,68,68,0.5)]'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      Team B
                    </button>
                  </div>
                </div>
              </div>

              {/* Sub-Filters: Minutes Dropdown + Player Dropdown */}
              <div className="flex items-center justify-between text-xs text-slate-400 mb-3 px-1">
                <div className="font-semibold text-white/90">
                  Football Match Position Heatmap — {heatmapTeam === 'a' ? 'Team A' : 'Team B'}
                </div>

                <div className="flex items-center gap-2">
                  {/* Time Range Dropdown */}
                  <div className="relative">
                    <button
                      onClick={() => setTimeDropdownOpen(!timeDropdownOpen)}
                      className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-white font-medium text-xs shadow-sm"
                    >
                      <span>{timeWindow}</span>
                      <ChevronDown size={12} className="text-slate-400" />
                    </button>
                    {timeDropdownOpen && (
                      <div className="absolute right-0 mt-1 w-32 glass bg-[#070d1a]/95 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl z-30 py-1 text-xs">
                        {['Minutes 0-15', 'Minutes 15-30', 'Minutes 30-45', 'Minutes 45-60', 'Full Match'].map(t => (
                          <button
                            key={t}
                            onClick={() => { setTimeWindow(t); setTimeDropdownOpen(false); }}
                            className="w-full text-left px-3 py-1.5 hover:bg-white/10 text-slate-200"
                          >
                            {t}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Player Dropdown */}
                  <div className="relative">
                    <button
                      onClick={() => setPlayerDropdownOpen(!playerDropdownOpen)}
                      className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-white font-medium text-xs shadow-sm"
                    >
                      <span>Player {selectedPlayerId}</span>
                      <ChevronDown size={12} className="text-slate-400" />
                    </button>
                    {playerDropdownOpen && (
                      <div className="absolute right-0 mt-1 w-28 glass bg-[#070d1a]/95 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl z-30 py-1 text-xs">
                        {FORMATION_433.map(p => (
                          <button
                            key={p.id}
                            onClick={() => { setSelectedPlayerId(p.id); setPlayerDropdownOpen(false); }}
                            className="w-full text-left px-3 py-1.5 hover:bg-white/10 text-slate-200"
                          >
                            Player {p.id}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Pitch + Heatmap + Player Vertical Strip */}
              <div className="flex gap-3">
                {/* Pitch Canvas Area with Coordinates */}
                <div className="flex-1 relative rounded-xl border border-white/10 bg-[#0d1f12] overflow-hidden p-2 flex flex-col justify-between" style={{ minHeight: 310 }}>
                  {/* SVG Football Pitch Markings with Dynamic Thermal Contours */}
                  <svg viewBox="0 0 105 68" className="w-full h-full absolute inset-0">
                    {/* Grass background */}
                    <rect x="0" y="0" width="105" height="68" fill="#0c1f14" />
                    
                    {/* Pitch Boundary */}
                    <rect x="5" y="4" width="95" height="60" fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="0.8" />
                    {/* Halfway line */}
                    <line x1="52.5" y1="4" x2="52.5" y2="64" stroke="rgba(255,255,255,0.18)" strokeWidth="0.8" />
                    {/* Center circle */}
                    <circle cx="52.5" cy="34" r="9.15" fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="0.8" />
                    <circle cx="52.5" cy="34" r="0.6" fill="rgba(255,255,255,0.3)" />

                    {/* Left Penalty Area */}
                    <rect x="5" y="16" width="16.5" height="36" fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="0.8" />
                    <rect x="5" y="24" width="5.5" height="20" fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="0.8" />

                    {/* Right Penalty Area */}
                    <rect x="83.5" y="16" width="16.5" height="36" fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="0.8" />
                    <rect x="94.5" y="24" width="5.5" height="20" fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="0.8" />

                    {/* Dynamic Smooth Thermal Density Contours Matching Reference Image 2 */}
                    <g opacity="0.9">
                      {/* Outer Cyan Ring */}
                      <path
                        d="M 18,36 C 24,18 42,20 54,28 C 66,36 60,54 44,52 C 28,50 14,48 18,36 Z"
                        fill="rgba(0, 212, 255, 0.25)"
                        filter="blur(3.5px)"
                      />
                      {/* Mid Emerald Ring */}
                      <path
                        d="M 22,35 C 26,22 40,24 49,30 C 58,36 54,48 42,47 C 30,45 18,44 22,35 Z"
                        fill="rgba(34, 197, 94, 0.55)"
                        filter="blur(2.5px)"
                      />
                      {/* Yellow Core Ring */}
                      <path
                        d="M 25,34 C 28,26 38,27 43,32 C 48,36 45,43 37,42 C 30,41 22,40 25,34 Z"
                        fill="rgba(234, 179, 8, 0.78)"
                        filter="blur(1.8px)"
                      />
                      {/* Deep Crimson Hotspot Center */}
                      <path
                        d="M 28,33 C 31,29 36,30 39,33 C 42,36 39,40 33,39 C 29,38 27,37 28,33 Z"
                        fill="rgba(239, 68, 68, 0.96)"
                        filter="blur(1.2px)"
                      />
                    </g>
                  </svg>

                  {/* Floating Inspection Tooltip */}
                  <div className="absolute top-1/2 left-[58%] -translate-y-1/2 glass bg-black/80 px-3 py-2 rounded-xl border border-white/20 text-[10px] text-white shadow-xl pointer-events-none">
                    <div className="font-mono text-slate-300">X: 20.06</div>
                    <div className="font-mono text-slate-300">Y: 23.72</div>
                    <div className="font-mono text-[#00d4ff] font-bold mt-0.5">Position Density: 1.35</div>
                  </div>

                  {/* Y Axis: Pitch Width (meters) */}
                  <div className="relative z-10 text-[9px] text-slate-400 flex flex-col justify-between h-full font-mono pl-1 py-1 pointer-events-none">
                    <span>60</span>
                    <span>40</span>
                    <span>20</span>
                    <span>0</span>
                  </div>

                  {/* X Axis: Pitch Length (meters) */}
                  <div className="relative z-10 text-[9px] text-slate-400 flex justify-between font-mono px-6 pt-1 pointer-events-none">
                    <span>0</span>
                    <span>20</span>
                    <span>40</span>
                    <span>60</span>
                    <span>80</span>
                    <span>100</span>
                  </div>
                </div>

                {/* Vertical Player List Strip */}
                <div className="w-12 flex flex-col items-center gap-2 overflow-y-auto max-h-[310px] pr-1 py-1">
                  <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-1">
                    Player
                  </span>
                  {FORMATION_433.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => setSelectedPlayerId(p.id)}
                      title={`#${p.id} ${p.name}`}
                      className={`relative w-8 h-8 rounded-full overflow-hidden border-2 transition-all duration-200 ${
                        selectedPlayerId === p.id
                          ? 'border-[#00d4ff] scale-110 shadow-[0_0_12px_rgba(0,212,255,0.6)] ring-2 ring-[#00d4ff]/40'
                          : 'border-white/20 opacity-70 hover:opacity-100 hover:scale-105'
                      }`}
                    >
                      <img
                        src={heatmapTeam === 'a' ? TEAM_A_AVATAR : TEAM_B_AVATAR}
                        alt={p.name}
                        className="w-full h-full object-cover"
                      />
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Side-by-side Mini Heatmaps with Goal-Scoring Zone */}
            <div className="grid grid-cols-2 gap-4">
              {/* Team A Mini Heatmap */}
              <div className="glass tactics-card-glass p-3.5 rounded-2xl border border-white/10 shadow-md">
                <div className="text-xs font-bold text-slate-300 mb-2 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#0070f3]" /> Team A
                </div>
                <div className="pitch-bg h-28 rounded-xl relative overflow-hidden border border-white/10">
                  {/* Heatmap blur */}
                  <div
                    className="absolute inset-0"
                    style={{
                      background: `radial-gradient(circle at 35% 65%, rgba(239,68,68,0.85) 0%, rgba(234,179,8,0.5) 25%, rgba(0,212,255,0.2) 50%, transparent 75%)`,
                    }}
                  />
                  {/* Goal-Scoring Zone Box */}
                  <div className="absolute right-2 top-2 bottom-2 w-20 border border-amber-400/50 bg-amber-400/10 rounded-lg flex flex-col items-center justify-center text-center p-1">
                    <span className="text-[9px] text-amber-300 font-bold uppercase tracking-tight leading-tight">
                      Goal-scoring Zone
                    </span>
                    <div className="flex items-center gap-1 text-[11px] text-white font-mono font-bold mt-1">
                      <Target size={11} className="text-amber-400" /> 36
                    </div>
                  </div>
                </div>
              </div>

              {/* Team B Mini Heatmap */}
              <div className="glass tactics-card-glass p-3.5 rounded-2xl border border-white/10 shadow-md">
                <div className="text-xs font-bold text-slate-300 mb-2 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#ef4444]" /> Team B
                </div>
                <div className="pitch-bg h-28 rounded-xl relative overflow-hidden border border-white/10">
                  {/* Heatmap blur */}
                  <div
                    className="absolute inset-0"
                    style={{
                      background: `radial-gradient(circle at 38% 60%, rgba(239,68,68,0.85) 0%, rgba(234,179,8,0.5) 25%, rgba(0,212,255,0.2) 50%, transparent 75%)`,
                    }}
                  />
                  {/* Goal-Scoring Zone Box */}
                  <div className="absolute right-2 top-2 bottom-2 w-20 border border-amber-400/50 bg-amber-400/10 rounded-lg flex flex-col items-center justify-center text-center p-1">
                    <span className="text-[9px] text-amber-300 font-bold uppercase tracking-tight leading-tight">
                      Goal-scoring Zone
                    </span>
                    <div className="flex items-center gap-1 text-[11px] text-white font-mono font-bold mt-1">
                      <Target size={11} className="text-amber-400" /> 12
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* ========================================================
              RIGHT COLUMN (6 cols): Passing Network + Match Momentum
              ======================================================== */}
          <div className="xl:col-span-6 flex flex-col gap-6">
            {/* Passing Network Card */}
            <div className="glass tactics-card-glass p-5 rounded-2xl border border-white/10 shadow-xl relative">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Network size={15} className="text-[#00d4ff]" />
                  <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                    Passing Network — Team A
                  </h2>
                </div>
              </div>

              {/* Pitch with 4-3-3 Passing Network */}
              <div className="relative pitch-bg rounded-xl border border-white/10 h-72 overflow-hidden">
                <svg viewBox="0 0 100 100" className="w-full h-full absolute inset-0">
                  {/* Pitch outline */}
                  <rect x="5" y="5" width="90" height="90" fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="0.8" />
                  <line x1="50" y1="5" x2="50" y2="95" stroke="rgba(255,255,255,0.15)" strokeWidth="0.8" />
                  <circle cx="50" cy="50" r="14" fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="0.8" />

                  {/* Passing Arrows / Lines */}
                  {/* Standard Blue Passes */}
                  <line x1="10" y1="50" x2="25" y2="38" stroke="#0070f3" strokeWidth="2.5" opacity="0.75" />
                  <line x1="10" y1="50" x2="25" y2="62" stroke="#0070f3" strokeWidth="2.5" opacity="0.75" />
                  <line x1="28" y1="18" x2="45" y2="50" stroke="#0070f3" strokeWidth="2.2" opacity="0.75" />
                  <line x1="25" y1="38" x2="45" y2="50" stroke="#0070f3" strokeWidth="3" opacity="0.85" />
                  <line x1="25" y1="62" x2="45" y2="50" stroke="#0070f3" strokeWidth="3" opacity="0.85" />
                  <line x1="28" y1="82" x2="45" y2="50" stroke="#0070f3" strokeWidth="2.2" opacity="0.75" />
                  <line x1="45" y1="50" x2="55" y2="30" stroke="#0070f3" strokeWidth="2.8" opacity="0.85" />
                  <line x1="45" y1="50" x2="62" y2="70" stroke="#0070f3" strokeWidth="3.2" opacity="0.85" />
                  <line x1="55" y1="30" x2="78" y2="20" stroke="#0070f3" strokeWidth="2" opacity="0.7" />
                  <line x1="55" y1="30" x2="82" y2="50" stroke="#0070f3" strokeWidth="2.5" opacity="0.8" />

                  {/* Highlighted Yellow Key Pass Links (Player 7 Gareth Best) */}
                  <line x1="62" y1="70" x2="82" y2="50" stroke="#eab308" strokeWidth="3.5" opacity="0.95" />
                  <line x1="62" y1="70" x2="78" y2="80" stroke="#eab308" strokeWidth="3.5" opacity="0.95" />

                  {/* Player Nodes */}
                  {FORMATION_433.map((p) => {
                    const isHighlighted = p.id === selectedPlayerId
                    return (
                      <g key={p.id} onClick={() => setSelectedPlayerId(p.id)} className="cursor-pointer">
                        <circle
                          cx={p.x}
                          cy={p.y}
                          r={isHighlighted ? 4.8 : 3.8}
                          fill="#070d1a"
                          stroke={isHighlighted ? '#eab308' : '#00d4ff'}
                          strokeWidth={isHighlighted ? 1.8 : 1.2}
                        />
                        <text
                          x={p.x}
                          y={p.y + 1}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fill="white"
                          fontSize="3"
                          fontWeight="bold"
                        >
                          {p.id}
                        </text>
                      </g>
                    )
                  })}
                </svg>

                {/* Floating Inspector Card for Selected Player */}
                <div className="absolute top-3 right-3 glass bg-black/85 p-3 rounded-xl border border-white/20 text-xs shadow-2xl w-44">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-8 h-8 rounded-full overflow-hidden border border-[#00d4ff]">
                      <img src={TEAM_A_AVATAR} alt={selectedPlayer.name} className="w-full h-full object-cover" />
                    </div>
                    <div>
                      <div className="text-[10px] text-slate-400">Player {selectedPlayer.id}</div>
                      <div className="font-bold text-white text-xs truncate">{selectedPlayer.name}</div>
                    </div>
                  </div>

                  <div className="space-y-1 text-[11px] text-slate-300 font-mono mb-2">
                    <div className="flex justify-between">
                      <span>Total Passes:</span>
                      <span className="font-bold text-white">28/29</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Received:</span>
                      <span className="font-bold text-white">31</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Pass accuracy:</span>
                      <span className="font-bold text-[#00d4ff]">97%</span>
                    </div>
                  </div>

                  {/* Circular Gauge */}
                  <div className="flex items-center justify-center gap-2 pt-1 border-t border-white/10">
                    <div className="relative w-9 h-9">
                      <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
                        <circle cx="18" cy="18" r="14" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="3" />
                        <circle
                          cx="18"
                          cy="18"
                          r="14"
                          fill="none"
                          stroke="#eab308"
                          strokeWidth="3"
                          strokeDasharray="87, 100"
                          strokeLinecap="round"
                        />
                      </svg>
                      <span className="absolute inset-0 flex items-center justify-center text-[9px] font-bold text-white">
                        87%
                      </span>
                    </div>
                    <span className="text-[9px] text-slate-400 font-semibold leading-tight">
                      Total Passes
                    </span>
                  </div>
                </div>

                {/* Pass Length Filter Checklist (Bottom Left) */}
                <div className="absolute bottom-2 left-2 glass bg-black/80 px-2.5 py-2 rounded-xl border border-white/15 text-[10px] text-slate-300">
                  <div className="font-bold text-white mb-1 uppercase tracking-wider text-[9px]">
                    Pass Length
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={passFilter.short}
                        onChange={() => setPassFilter({ ...passFilter, short: !passFilter.short })}
                        className="accent-[#00d4ff] rounded"
                      />
                      <span>Short</span>
                    </label>
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={passFilter.medium}
                        onChange={() => setPassFilter({ ...passFilter, medium: !passFilter.medium })}
                        className="accent-[#00d4ff] rounded"
                      />
                      <span>Medium</span>
                    </label>
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={passFilter.long}
                        onChange={() => setPassFilter({ ...passFilter, long: !passFilter.long })}
                        className="accent-[#00d4ff] rounded"
                      />
                      <span>Long</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>

            {/* Match Momentum Card */}
            <div className="glass tactics-card-glass p-5 rounded-2xl border border-white/10 shadow-xl relative flex flex-col gap-4">
              {/* Header + AI Insight Toggle */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <TrendingUp size={15} className="text-[#00d4ff]" />
                  <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                    Match Momentum
                  </h2>
                </div>

                {/* AI Insight Box (Top Right Pill) */}
                <div className="glass bg-[#00d4ff]/10 border border-[#00d4ff]/30 p-2.5 rounded-xl max-w-xs shadow-md">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <span className="text-[10px] font-black text-[#00d4ff] uppercase tracking-wider flex items-center gap-1">
                      <Sparkles size={11} /> AI INSIGHT:
                    </span>
                    <button
                      onClick={() => setAiInsightOpen(!aiInsightOpen)}
                      className="text-slate-400 hover:text-white"
                    >
                      {aiInsightOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                    </button>
                  </div>
                  {aiInsightOpen && (
                    <p className="text-[10px] text-slate-300 leading-relaxed font-medium">
                      Team A's vertical passing frequency increased by 20% after the 35th-minute stabilization, directly contributing to the late momentum shift.
                    </p>
                  )}
                </div>
              </div>

              {/* Momentum Wave Graphic (+4 to -3) with Pinned Event Icons */}
              <div className="relative h-44 w-full pt-2">
                <svg viewBox="0 0 100 50" className="w-full h-full">
                  {/* Zero Center Line */}
                  <line x1="5" y1="25" x2="95" y2="25" stroke="rgba(255,255,255,0.2)" strokeWidth="0.5" strokeDasharray="1,1" />

                  {/* Team A Momentum Wave (Positive: 0 to -4 y scale) */}
                  <path
                    d="M 5,25 Q 15,22 25,23 Q 32,15 40,8 Q 50,7 60,18 Q 70,26 80,32 Q 90,16 95,25 L 95,25 L 5,25 Z"
                    fill="rgba(0, 212, 255, 0.25)"
                    stroke="#00d4ff"
                    strokeWidth="1.2"
                  />

                  {/* Team B Momentum Wave (Negative: 0 to +3 y scale) */}
                  <path
                    d="M 5,25 Q 20,29 35,32 Q 50,30 65,37 Q 75,44 85,38 Q 92,28 95,25 L 95,25 L 5,25 Z"
                    fill="rgba(239, 68, 68, 0.25)"
                    stroke="#ef4444"
                    strokeWidth="1.2"
                  />

                  {/* Event Markers along Timeline */}
                  {/* 10' Sub */}
                  <circle cx="15" cy="22" r="1.5" fill="#f59e0b" />
                  {/* 25' Yellow Card */}
                  <circle cx="30" cy="17" r="1.5" fill="#eab308" />
                  {/* 35' Goal */}
                  <circle cx="40" cy="8" r="2.2" fill="#22c55e" stroke="#fff" strokeWidth="0.5" />
                  {/* 60' Sub */}
                  <circle cx="65" cy="37" r="1.5" fill="#3b82f6" />
                  {/* 75' Card */}
                  <circle cx="85" cy="38" r="1.5" fill="#ef4444" />
                </svg>

                {/* Y-Axis Labels */}
                <div className="absolute left-0 top-0 bottom-0 flex flex-col justify-between text-[9px] font-mono text-slate-500 py-1">
                  <span>+4</span>
                  <span>+2</span>
                  <span>0</span>
                  <span>-2</span>
                  <span>-3</span>
                </div>

                {/* X-Axis Minute Labels */}
                <div className="flex justify-between text-[8px] font-mono text-slate-400 px-6 pt-1">
                  <span>0'</span>
                  <span>5'</span>
                  <span>10'</span>
                  <span>15'</span>
                  <span>20'</span>
                  <span>25'</span>
                  <span>30'</span>
                  <span>35'</span>
                  <span>40'</span>
                  <span>45'</span>
                  <span>50'</span>
                  <span>65'</span>
                  <span>60'</span>
                  <span>75'</span>
                  <span>90'</span>
                </div>
              </div>

              {/* Interactive Scrubber & Timeline Bar */}
              <div className="flex items-center gap-3 pt-2 border-t border-white/10">
                <button
                  onClick={() => setIsPlayingScrubber(!isPlayingScrubber)}
                  className="w-8 h-8 rounded-full bg-[#00d4ff]/15 hover:bg-[#00d4ff]/30 text-[#00d4ff] flex items-center justify-center transition-colors shadow-sm"
                >
                  {isPlayingScrubber ? <Pause size={14} /> : <Play size={14} className="ml-0.5" />}
                </button>

                <div className="flex-1 relative">
                  <input
                    type="range"
                    min="0"
                    max="90"
                    value={scrubberMin}
                    onChange={(e) => setScrubberMin(Number(e.target.value))}
                    className="w-full accent-[#00d4ff] h-1.5 bg-white/10 rounded-lg cursor-pointer"
                  />
                  <span
                    className="absolute -top-6 px-1.5 py-0.5 rounded bg-[#00d4ff] text-slate-950 text-[9px] font-mono font-bold -translate-x-1/2 shadow-md"
                    style={{ left: `${(scrubberMin / 90) * 100}%` }}
                  >
                    {scrubberMin}'
                  </span>
                </div>

                {/* Legend */}
                <div className="flex items-center gap-3 text-xs font-semibold pl-2">
                  <span className="flex items-center gap-1 text-[#00d4ff]">
                    <span className="w-2 h-2 rounded-full bg-[#00d4ff]" /> Team A
                  </span>
                  <span className="flex items-center gap-1 text-[#ef4444]">
                    <span className="w-2 h-2 rounded-full bg-[#ef4444]" /> Team B
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
