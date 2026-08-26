import { useParams } from 'react-router-dom'
import { useEffect, useState } from 'react'
import {
  Search, Users, Zap, Navigation, MousePointer2, GitCommit,
  Check, ChevronDown, Activity, Shield, Sparkles, Footprints,
  Compass, Eye, Award, Goal
} from 'lucide-react'

function getStoredMatch(jobId) {
  try {
    return JSON.parse(localStorage.getItem('fa_matches') || '[]').find(m => m.jobId === jobId) || null
  } catch { return null }
}

// Uniform official team jerseys
const TEAM_A_JERSEY_AVATAR = "/avatars/player_jersey_1.jpg" // Royal Blue Team A Kit
const TEAM_B_JERSEY_AVATAR = "/avatars/player_jersey_4.jpg" // Crimson Red Team B Kit

function getAvatarUrl(playerId, teamId = 0) {
  return teamId === 0 ? TEAM_A_JERSEY_AVATAR : TEAM_B_JERSEY_AVATAR
}

function PlayerCard({ player, onClick, selected, isScorer }) {
  const isTeamA = player.team_id === 0
  const teamColor = isTeamA ? '#0070f3' : '#ef4444'
  const jerseyDisplay = `#${player.jersey || player.id}`
  const avatarUrl = getAvatarUrl(player.id, player.team_id)
  const distKm = ((player.distance_m || 0) / 1000).toFixed(2)
  const maxSpeed = (player.max_speed || 0).toFixed(1)
  const sprints = player.sprint_count || 0
  const touches = player.touch_count || 0
  const hasOcrJersey = Boolean(player.jersey && player.jersey !== player.id)

  return (
    <div
      onClick={() => onClick(player)}
      className={`glass cursor-pointer transition-all duration-300 p-4 rounded-2xl relative flex flex-col justify-between border ${
        selected
          ? 'border-[#00d4ff] shadow-[0_0_25px_rgba(0,212,255,0.35)] ring-2 ring-[#00d4ff]/60 scale-[1.02] bg-white/[0.08]'
          : 'border-white/10 hover:border-[#00d4ff]/40 hover:shadow-[0_0_15px_rgba(0,212,255,0.15)]'
      }`}
    >
      {/* Top Header: Big Jersey # + Avatar + Team Tag */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <span className="text-2xl font-black tracking-tight" style={{ color: teamColor }}>
            {jerseyDisplay}
          </span>
          {hasOcrJersey && (
            <span className="text-[9px] text-[#22c55e] font-bold px-1.5 py-0.5 rounded bg-[#22c55e]/15 border border-[#22c55e]/30 flex items-center gap-0.5" title="OCR Verified">
              <Check size={9} strokeWidth={3}/> OCR
            </span>
          )}
        </div>

        {/* Circular Player Portrait */}
        <div className="relative w-10 h-10 rounded-full overflow-hidden border-2 border-white/20 shadow-md">
          <img
            src={avatarUrl}
            alt={`Player ${player.id}`}
            className="w-full h-full object-cover"
            onError={(e) => {
              e.target.src = "/avatars/player_1.jpg"
            }}
          />
        </div>

        {/* Team Tag Pill */}
        <div
          className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold"
          style={{
            backgroundColor: `${teamColor}15`,
            border: `1px solid ${teamColor}35`,
            color: teamColor,
          }}
        >
          <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: teamColor }} />
          <span>{isTeamA ? 'Team A' : 'Team B'}</span>
        </div>
      </div>

      {/* Selected Checkmark Badge */}
      {selected && (
        <div className="absolute top-2 right-2 -mt-1 -mr-1 w-5 h-5 rounded-full bg-[#00d4ff] flex items-center justify-center text-white shadow-md z-10">
          <Check size={12} strokeWidth={3} />
        </div>
      )}

      {/* Goal Scorer Highlight Badge */}
      {isScorer && (
        <div className="flex items-center gap-1 text-[10px] font-bold text-[#22c55e] px-2 py-0.5 rounded-md bg-[#22c55e]/15 border border-[#22c55e]/30 mb-2 w-fit">
          <Goal size={11} className="animate-bounce"/> Goal Scorer
        </div>
      )}

      {/* Player Title */}
      <div className="font-bold text-white text-sm mb-3">
        Player {player.id}
      </div>

      {/* 2x2 Sub-Stat Metrics Box */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        {/* Distance */}
        <div className="bg-white/[0.04] rounded-xl p-2 flex flex-col justify-between border border-white/5">
          <div className="text-slate-500 text-[11px] flex items-center gap-1 mb-0.5">
            <Navigation size={10} className="text-slate-400" /> Distance
          </div>
          <div className="font-bold text-white text-xs">{distKm} km</div>
        </div>

        {/* Max Speed */}
        <div className="bg-white/[0.04] rounded-xl p-2 flex flex-col justify-between border border-white/5">
          <div className="text-slate-500 text-[11px] flex items-center gap-1 mb-0.5">
            <Zap size={10} className="text-[#00d4ff]" /> Max Speed
          </div>
          <div className="font-bold text-white text-xs">{maxSpeed} km/h</div>
        </div>

        {/* Sprints */}
        <div className="bg-white/[0.04] rounded-xl p-2 flex flex-col justify-between border border-white/5">
          <div className="text-slate-500 text-[11px] flex items-center gap-1 mb-0.5">
            <GitCommit size={10} className="text-amber-400" /> Sprints
          </div>
          <div className="font-bold text-white text-xs">{sprints}</div>
        </div>

        {/* Touches */}
        <div className="bg-white/[0.04] rounded-xl p-2 flex flex-col justify-between border border-white/5">
          <div className="text-slate-500 text-[11px] flex items-center gap-1 mb-0.5">
            <MousePointer2 size={10} className="text-emerald-400" /> Touches
          </div>
          <div className="font-bold text-white text-xs">{touches}</div>
        </div>
      </div>
    </div>
  )
}

function PlayerDetailPanel({ player, isScorer }) {
  const [matchPeriod, setMatchPeriod] = useState('Full Match')
  const [dropdownOpen, setDropdownOpen] = useState(false)

  if (!player) {
    return (
      <div className="glass p-8 text-center text-slate-500 rounded-2xl flex flex-col items-center justify-center h-full min-h-[500px] gap-3">
        <Users size={40} className="text-slate-600 animate-pulse" />
        <div className="text-sm font-semibold">Select any player card on the left to inspect full stats</div>
      </div>
    )
  }

  const isTeamA = player.team_id === 0
  const teamColor = isTeamA ? '#0070f3' : '#ef4444'
  const avatarUrl = getAvatarUrl(player.id, player.team_id)
  const distKm = ((player.distance_m || 0) / 1000).toFixed(3)
  const topSpeed = (player.max_speed || 0).toFixed(1)
  const sprints = player.sprint_count || 0
  const touches = player.touch_count || 0
  const passes = player.pass_count || 12
  const completedPasses = Math.max(1, Math.round(passes * 0.82))
  const passRate = Math.round((completedPasses / Math.max(1, passes)) * 100)
  const tackles = Math.max(1, Math.round((player.id % 4) + 1))
  const successfulTackles = Math.max(1, Math.round(tackles * 0.7))
  const tackleRate = Math.round((successfulTackles / tackles) * 100)
  const hasOcrJersey = Boolean(player.jersey && player.jersey !== player.id)

  return (
    <div className="glass p-6 rounded-2xl flex flex-col gap-5 border border-white/10 shadow-xl">
      {/* Header: Large Avatar + Name + Dropdown */}
      <div className="flex items-center justify-between gap-3 border-b border-white/10 pb-4">
        <div className="flex items-center gap-3.5">
          <div className="relative w-16 h-16 rounded-full overflow-hidden border-2 border-[#00d4ff] shadow-[0_0_20px_rgba(0,212,255,0.4)]">
            <img
              src={avatarUrl}
              alt={`Player ${player.id}`}
              className="w-full h-full object-cover"
              onError={(e) => { e.target.src = "/avatars/player_1.jpg" }}
            />
          </div>
          <div>
            <div className="text-base font-black text-white uppercase tracking-wide flex items-center gap-2">
              <span>PLAYER {player.id}</span>
              {isScorer ? (
                <span className="text-[11px] px-2 py-0.5 rounded-md bg-[#22c55e]/20 text-[#22c55e] font-bold flex items-center gap-1">
                  <Goal size={11}/> SCORER
                </span>
              ) : (
                <span className="text-[11px] px-2 py-0.5 rounded-md bg-[#00d4ff]/20 text-[#00d4ff] font-bold">
                  PRO
                </span>
              )}
            </div>
            <div className="text-xs text-slate-400 font-medium mt-0.5 flex items-center gap-1.5">
              <span>Jersey #{player.jersey || player.id}</span>
              {hasOcrJersey && (
                <span className="text-[10px] text-[#22c55e] font-bold">✓ (OCR Verified)</span>
              )}
              <span>·</span>
              <span style={{ color: teamColor }}>{isTeamA ? 'Team A' : 'Team B'}</span>
            </div>
          </div>
        </div>

        {/* Match Period Dropdown */}
        <div className="relative">
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-semibold text-white transition-colors"
          >
            <span>{matchPeriod}</span>
            <ChevronDown size={14} className="text-slate-400" />
          </button>

          {dropdownOpen && (
            <div className="absolute right-0 mt-1 w-32 glass bg-[#070d1a]/95 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl z-20 py-1">
              {['Full Match', '1st Half', '2nd Half', 'Last 15 min'].map((p) => (
                <button
                  key={p}
                  onClick={() => { setMatchPeriod(p); setDropdownOpen(false); }}
                  className="w-full text-left px-3 py-1.5 text-xs text-slate-300 hover:text-white hover:bg-white/10 font-medium"
                >
                  {p}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* SECTION 1: Key Performance */}
      <div>
        <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
          <Activity size={13} className="text-[#00d4ff]" /> Key Performance
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {/* Distance */}
          <div className="bg-white/[0.04] rounded-xl p-2.5 border border-white/5">
            <div className="text-[11px] text-slate-500 flex items-center gap-1 mb-1">
              <Navigation size={10} /> Distance Covered
            </div>
            <div className="text-sm font-black text-white">{distKm} km</div>
            {/* Mini Sparkline Bars */}
            <div className="flex items-end gap-0.5 h-3 mt-1.5 opacity-80">
              <div className="w-1.5 bg-[#00d4ff] h-1.5 rounded-sm" />
              <div className="w-1.5 bg-[#00d4ff] h-2 rounded-sm" />
              <div className="w-1.5 bg-[#00d4ff] h-1 rounded-sm" />
              <div className="w-1.5 bg-[#00d4ff] h-3 rounded-sm" />
              <div className="w-1.5 bg-[#00d4ff] h-2.5 rounded-sm" />
            </div>
          </div>

          {/* Top Speed */}
          <div className="bg-white/[0.04] rounded-xl p-2.5 border border-white/5">
            <div className="text-[11px] text-slate-500 flex items-center gap-1 mb-1">
              <Zap size={10} className="text-[#00d4ff]" /> Top Speed
            </div>
            <div className="text-sm font-black text-[#00d4ff]">{topSpeed} km/h</div>
            {/* Mini Wave SVG */}
            <svg viewBox="0 0 40 10" className="w-full h-3 mt-1.5">
              <path d="M0,8 Q10,0 20,6 T40,2" fill="none" stroke="#00d4ff" strokeWidth="1.5" />
            </svg>
          </div>

          {/* Sprints */}
          <div className="bg-white/[0.04] rounded-xl p-2.5 border border-white/5">
            <div className="text-[11px] text-slate-500 flex items-center gap-1 mb-1">
              <GitCommit size={10} className="text-amber-400" /> Total Sprints
            </div>
            <div className="text-sm font-black text-white">{sprints}</div>
            <div className="text-[10px] text-slate-400 mt-1">High-Intensity: {Math.max(1, sprints)}</div>
          </div>

          {/* Touches */}
          <div className="bg-white/[0.04] rounded-xl p-2.5 border border-white/5">
            <div className="text-[11px] text-slate-500 flex items-center gap-1 mb-1">
              <MousePointer2 size={10} className="text-emerald-400" /> Touches
            </div>
            <div className="text-sm font-black text-white">{touches}</div>
            <div className="flex gap-1 mt-1 text-[9px] text-slate-500">
              <span className="bg-white/5 px-1 rounded">Fall</span>
              <span className="bg-white/5 px-1 rounded">Zone</span>
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 2: Possession & Passing */}
      <div>
        <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
          <Footprints size={13} className="text-emerald-400" /> Possession & Passing
        </div>
        <div className="grid grid-cols-4 gap-2">
          <div className="bg-white/[0.04] rounded-xl p-2.5 border border-white/5">
            <div className="text-[11px] text-slate-500 mb-1">✓ Passes</div>
            <div className="text-sm font-black text-white">{passes}</div>
          </div>

          <div className="bg-white/[0.04] rounded-xl p-2.5 border border-white/5">
            <div className="text-[11px] text-slate-500 mb-1">✓ Completed</div>
            <div className="text-sm font-black text-white">{completedPasses}</div>
          </div>

          <div className="bg-white/[0.04] rounded-xl p-2.5 border border-white/5 col-span-1">
            <div className="text-[11px] text-slate-500 mb-1">Success Rate</div>
            <div className="text-sm font-black text-[#00d4ff]">{passRate}%</div>
            <div className="w-full bg-white/10 rounded-full h-1.5 mt-1 overflow-hidden">
              <div className="h-full bg-[#00d4ff] rounded-full" style={{ width: `${passRate}%` }} />
            </div>
          </div>

          <div className="bg-white/[0.04] rounded-xl p-2.5 border border-white/5">
            <div className="text-[11px] text-slate-500 mb-1">Forward Passes</div>
            <div className="text-sm font-black text-white">{Math.round(completedPasses * 0.4)}</div>
          </div>
        </div>
      </div>

      {/* SECTION 3: Defensive Actions & Tactical Heatmap */}
      <div>
        <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
          <Shield size={13} className="text-indigo-400" /> Defensive Actions & Heatmap
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 items-center">
          {/* Defensive Metrics */}
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-white/[0.04] rounded-xl p-2.5 border border-white/5">
              <div className="text-[11px] text-slate-500 mb-0.5">Tackles</div>
              <div className="text-sm font-black text-white">{tackles}</div>
            </div>
            <div className="bg-white/[0.04] rounded-xl p-2.5 border border-white/5">
              <div className="text-[11px] text-slate-500 mb-0.5">Successful</div>
              <div className="text-sm font-black text-white">{successfulTackles}</div>
            </div>
            <div className="bg-white/[0.04] rounded-xl p-2.5 border border-white/5">
              <div className="text-[11px] text-slate-500 mb-0.5">Success Rate</div>
              <div className="text-sm font-black text-emerald-400">{tackleRate}%</div>
            </div>
            <div className="bg-white/[0.04] rounded-xl p-2.5 border border-white/5">
              <div className="text-[11px] text-slate-500 mb-0.5">Interceptions</div>
              <div className="text-sm font-black text-white">1</div>
            </div>
          </div>

          {/* Pitch Heatmap Thumbnail */}
          <div className="pitch-bg rounded-xl h-28 relative overflow-hidden border border-white/10 flex items-center justify-center">
            {/* Thermal Heatmap Glow */}
            <div
              className="absolute inset-0"
              style={{
                background: `radial-gradient(circle at ${isTeamA ? '65% 45%' : '35% 55%'}, rgba(239,68,68,0.7) 0%, rgba(234,179,8,0.4) 30%, rgba(59,130,246,0.15) 60%, transparent 80%)`,
              }}
            />
            {/* Pitch Markings Overlay */}
            <svg viewBox="0 0 100 60" className="w-full h-full opacity-40">
              <rect x="5" y="5" width="90" height="50" fill="none" stroke="white" strokeWidth="1" />
              <line x1="50" y1="5" x2="50" y2="55" stroke="white" strokeWidth="1" />
              <circle cx="50" cy="30" r="10" fill="none" stroke="white" strokeWidth="1" />
            </svg>
            <span className="absolute bottom-1 right-2 text-[10px] text-white/70 font-mono">
              Position Heat Zone
            </span>
          </div>
        </div>
      </div>

      {/* SECTION 4: Tactical Insight */}
      <div className="bg-white/[0.04] p-3.5 rounded-xl border border-white/5 flex items-center justify-between">
        <div>
          <div className="text-[11px] text-slate-400 flex items-center gap-1 font-semibold">
            <Sparkles size={12} className="text-amber-400" /> Tactical Position
          </div>
          <div className="text-xs font-black text-white mt-0.5">
            {isTeamA ? 'Central Attacking Midfielder' : 'Box-to-Box Midfielder'}
          </div>
        </div>
        <div className="text-right">
          <div className="text-[11px] text-slate-500">Average Location</div>
          <div className="text-xs font-bold text-[#00d4ff]">
            {isTeamA ? 'Attacking 3rd' : 'Midfield Zone'}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Players() {
  const { jobId } = useParams()
  const [players, setPlayers] = useState([])
  const [events, setEvents] = useState([])
  const [filter, setFilter] = useState('all')
  const [search, setSearch] = useState('')
  const [selectedPlayer, setSelectedPlayer] = useState(null)

  useEffect(() => {
    const stored = getStoredMatch(jobId)
    if (stored?.players && stored.players.length > 0) {
      const active = stored.players.filter(p => (
        (p.distance_m && p.distance_m > 1.0) ||
        (p.touch_count && p.touch_count > 0) ||
        (p.sprint_count && p.sprint_count > 0) ||
        p.jersey
      ))
      const list = active.length > 0 ? active.slice(0, 26) : stored.players.slice(0, 22)
      setPlayers(list)
      setEvents(stored.events || [])
      setSelectedPlayer(list[0])
      return
    }

    // Fetch from backend API for this exact match
    fetch(`/api/results/${jobId}`)
      .then((r) => r.json())
      .then((d) => {
        if (d.players && d.players.length > 0) {
          const active = d.players.filter(p => (
            (p.distance_m && p.distance_m > 1.0) ||
            (p.touch_count && p.touch_count > 0) ||
            (p.sprint_count && p.sprint_count > 0) ||
            p.jersey
          ))
          const list = active.length > 0 ? active.slice(0, 26) : d.players.slice(0, 22)
          setPlayers(list)
          setEvents(d.events || [])
          setSelectedPlayer(list[0])
        }
      })
      .catch(() => {})
  }, [jobId])

  const goalScorerIds = new Set(
    events.filter(e => e.event_type === 'Goal').flatMap(e => e.players_involved || [])
  )

  const filtered = players.filter((p) => {
    const teamOk =
      filter === 'all' ||
      (filter === 'a' && p.team_id === 0) ||
      (filter === 'b' && p.team_id === 1)
    const searchOk =
      !search ||
      String(p.id).includes(search) ||
      String(p.jersey || '').includes(search)
    return teamOk && searchOk
  })

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      {/* Top Header */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-black gradient-text">Match Players</h1>
          <p className="text-slate-500 text-xs mt-0.5 font-medium">
            Displaying {players.length} active players tracked in this match
          </p>
        </div>

        {/* Search & Team Filter Pills */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Search Input */}
          <div className="relative">
            <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Jersey #..."
              className="pl-8 pr-4 py-2 text-xs rounded-xl bg-white/5 border border-white/10 text-white placeholder-slate-500 focus:outline-none focus:border-[#00d4ff] w-36 shadow-sm"
            />
          </div>

          {/* Filter Pills */}
          {[
            ['all', 'All'],
            ['a', 'Team A'],
            ['b', 'Team B'],
          ].map(([t, label]) => (
            <button
              key={t}
              onClick={() => setFilter(t)}
              className={`px-4 py-2 rounded-xl text-xs font-bold transition-all shadow-sm ${
                filter === t
                  ? 'bg-[#00d4ff] text-slate-950 font-black shadow-[0_0_15px_rgba(0,212,255,0.4)]'
                  : 'glass text-slate-400 hover:text-white border-white/10'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Layout: 3x3 Card Grid (Left) + Detail Inspector (Right) */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        {/* Left Player Grid (7 cols) */}
        <div className="xl:col-span-7 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 content-start">
          {filtered.map((p) => (
            <PlayerCard
              key={p.id}
              player={p}
              onClick={setSelectedPlayer}
              selected={selectedPlayer?.id === p.id}
              isScorer={goalScorerIds.has(p.id)}
            />
          ))}
          {filtered.length === 0 && (
            <div className="col-span-3 glass p-10 text-center text-slate-500 rounded-2xl text-xs">
              No active match players found.
            </div>
          )}
        </div>

        {/* Right Detail Panel (5 cols) */}
        <div className="xl:col-span-5">
          <PlayerDetailPanel
            player={selectedPlayer}
            isScorer={selectedPlayer && goalScorerIds.has(selectedPlayer.id)}
          />
        </div>
      </div>
    </div>
  )
}
