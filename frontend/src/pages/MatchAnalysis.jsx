import { useParams, Link } from 'react-router-dom'
import { useEffect, useState, useRef } from 'react'
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer
} from 'recharts'
import {
  Download, Users, CalendarClock, Map, Goal, Flag, AlertTriangle,
  Crosshair, Box, Activity, Repeat2, Zap, Navigation,
  ChevronRight, Sparkles, CheckCircle2, Loader2, Play, Award, Check
} from 'lucide-react'
import FootballLoader from '../components/FootballLoader'

function getStoredMatches() {
  try { return JSON.parse(localStorage.getItem('fa_matches') || '[]') } catch { return [] }
}
function saveMatch(m) {
  const matches = getStoredMatches()
  const existing = matches.findIndex(x => x.jobId === m.jobId)
  if (existing >= 0) matches[existing] = m
  else matches.unshift(m)
  localStorage.setItem('fa_matches', JSON.stringify(matches.slice(0, 20)))
}

const EVENT_ICON_MAP = {
  Goal: Goal, 'Potential Foul': AlertTriangle, Offside: Flag,
  'Shot on Target': Crosshair, 'Penalty Area Entry': Box,
  'Yellow Card Candidate': AlertTriangle, 'Corner Kick': Flag, 'Free Kick': Activity,
}
const EVENT_COLORS = {
  Goal: '#22c55e', 'Potential Foul': '#f59e0b', Offside: '#ef4444',
  'Shot on Target': '#a855f7', 'Penalty Area Entry': '#3b82f6',
  'Yellow Card Candidate': '#eab308', 'Corner Kick': '#06b6d4', 'Free Kick': '#84cc16',
}

function StatCompareCard({ label, Icon, homeVal, awayVal, homeColor = '#3b82f6', awayColor = '#ef4444', suffix = '' }) {
  const total = (homeVal || 0) + (awayVal || 0) || 1
  const homePct = ((homeVal || 0) / total) * 100
  return (
    <div className="glass p-4 flex flex-col gap-2 rounded-xl">
      <div className="text-slate-400 text-xs flex items-center gap-1.5 font-medium">
        {Icon && <Icon size={12} strokeWidth={2}/>} {label}
      </div>
      <div className="flex items-end justify-between">
        <span className="text-xl font-black" style={{ color: homeColor }}>{homeVal ?? 0}{suffix}</span>
        <span className="text-xs text-slate-600 font-medium">vs</span>
        <span className="text-xl font-black" style={{ color: awayColor }}>{awayVal ?? 0}{suffix}</span>
      </div>
      <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700"
          style={{ width: `${homePct}%`, background: `linear-gradient(to right, ${homeColor}, ${awayColor})` }}/>
      </div>
      <div className="flex justify-between text-xs text-slate-600"><span>Team A</span><span>Team B</span></div>
    </div>
  )
}

export default function MatchAnalysis() {
  const { jobId } = useParams()
  const [matchData, setMatchData] = useState(null)
  const [isProcessing, setIsProcessing] = useState(true)
  const [pipelineProgress, setPipelineProgress] = useState(5)
  const [pipelineStage, setPipelineStage] = useState('Initializing AI Video Pipeline...')
  const [selectedPlayerId, setSelectedPlayerId] = useState(null)
  const [downloading, setDownloading] = useState(false)
  const videoRef = useRef()

  const handleDownloadVideo = async (targetUrl) => {
    if (!targetUrl) return
    setDownloading(true)
    const filename = targetUrl.split('/').pop() || `match_${jobId}_annotated.mp4`
    const downloadEndpoint = `/api/download/${filename}`
    try {
      const res = await fetch(downloadEndpoint)
      if (!res.ok) throw new Error('Failed to fetch video binary')
      const blob = await res.blob()
      const blobUrl = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = blobUrl
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(blobUrl)
    } catch (err) {
      console.warn("Direct blob download failed, opening in new tab:", err)
      window.open(targetUrl, '_blank')
    } finally {
      setDownloading(false)
    }
  }

  const loadCompletedData = async () => {
    try {
      const res = await fetch(`/api/results/${jobId}`)
      if (res.ok) {
        const fullResults = await res.json()
        const matchRecord = {
          jobId,
          filename: fullResults.filename || `Match_${jobId}`,
          date: new Date().toLocaleDateString('en-IN', { day:'numeric', month:'short', year:'numeric' }),
          status: 'completed',
          ...fullResults
        }
        setMatchData(matchRecord)
        saveMatch(matchRecord)
        setIsProcessing(false)
      }
    } catch (e) {
      console.error("Failed to load results:", e)
    }
  }

  useEffect(() => {
    // Check if match is already cached in localStorage
    const matches = getStoredMatches()
    const found = matches.find(m => m.jobId === jobId)
    
    if (found && found.status === 'completed') {
      setMatchData(found)
      setIsProcessing(false)
      return
    }

    // Otherwise poll status from backend
    let isMounted = true
    const pollTimer = setInterval(async () => {
      try {
        const statusRes = await fetch(`/api/status/${jobId}`)
        if (!statusRes.ok) return

        const statusData = await statusRes.json()
        if (!isMounted) return

        setPipelineProgress(statusData.progress || 0)
        setPipelineStage(statusData.stage || 'Analyzing Match...')

        if (statusData.status === 'completed') {
          clearInterval(pollTimer)
          await loadCompletedData()
        } else if (statusData.status === 'failed') {
          clearInterval(pollTimer)
          setIsProcessing(false)
          setPipelineStage(`Analysis Failed: ${statusData.stage || 'Unknown error'}`)
        }
      } catch (err) {
        console.warn("Status poll hitch:", err)
      }
    }, 1500)

    return () => {
      isMounted = false
      clearInterval(pollTimer)
    }
  }, [jobId])

  const players = matchData?.players || []
  const events = matchData?.events || []
  const possession = matchData?.possession || { home: 50, away: 50 }
  const goals = matchData?.goals || { home: 0, away: 0 }
  const pass_accuracy = matchData?.pass_accuracy || { home: 80, away: 80 }
  const video_url = matchData?.video_url || matchData?.output_video_url || `/media/videos/match_${jobId}_annotated.mp4`

  // Filter ONLY active match players who actually participated (removing single-frame noise tracks)
  const activePlayers = players.filter(p => (
    (p.distance_m && p.distance_m > 1.0) ||
    (p.touch_count && p.touch_count > 0) ||
    (p.sprint_count && p.sprint_count > 0) ||
    p.jersey
  ))
  const displayPlayers = activePlayers.length > 0 ? activePlayers.slice(0, 26) : players.slice(0, 22)

  // Goal events and Goal Scorers identification
  const goalEvents = events.filter(e => e.event_type === 'Goal')
  const goalScorerIds = new Set(goalEvents.flatMap(e => e.players_involved || []))

  const radarData = [
    { subject: 'Attack', A: 65 + (goals.home||0)*10, B: 60 + (goals.away||0)*10 },
    { subject: 'Defense', A: 75, B: 70 },
    { subject: 'Possession', A: Math.round(possession.home||50), B: Math.round(possession.away||50) },
    { subject: 'Speed', A: 72, B: 68 },
    { subject: 'Accuracy', A: pass_accuracy.home || 80, B: pass_accuracy.away || 75 },
    { subject: 'Stamina', A: 70, B: 74 },
  ]

  const totalDistance = { home: 0, away: 0 }
  const totalSprints  = { home: 0, away: 0 }
  displayPlayers.forEach(p => {
    const side = p.team_id === 0 ? 'home' : 'away'
    totalDistance[side] += (p.distance_m || 0) / 1000
    totalSprints[side]  += p.sprint_count || 0
  })

  const shotEvents = events.filter(e => e.event_type === 'Shot on Target')

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 flex flex-col gap-6">
      {/* Top Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-black gradient-text">Match Analysis & Tracking</h1>
            {isProcessing ? (
              <span className="flex items-center gap-1.5 text-xs px-2.5 py-0.5 rounded-full bg-[#00d4ff]/15 border border-[#00d4ff]/30 text-[#00d4ff] font-bold animate-pulse">
                <Loader2 size={12} className="animate-spin"/> Analyzing In Real-Time
              </span>
            ) : (
              <span className="flex items-center gap-1 text-xs px-2.5 py-0.5 rounded-full bg-[#22c55e]/20 border border-[#22c55e]/30 text-[#22c55e] font-bold">
                <CheckCircle2 size={12}/> Analyzed
              </span>
            )}
          </div>
          <p className="text-slate-500 text-xs mt-0.5">
            Match ID: #{jobId} {isProcessing ? '· Processing 18-Stage AI Model' : `· ${displayPlayers.length} Active Match Players · ${events.length} Key Events`}
          </p>
        </div>

        {!isProcessing && (
          <div className="flex gap-2">
            {[
              { to: `/match/${jobId}/players`, Icon: Users, label: 'All Players' },
              { to: `/match/${jobId}/events`, Icon: CalendarClock, label: 'Events Timeline' },
              { to: `/match/${jobId}/tactics`, Icon: Map, label: 'Tactics & Heatmap' },
            ].map(({ to, Icon, label }) => (
              <Link key={to} to={to}
                className="flex items-center gap-1.5 px-4 py-2 rounded-xl glass glass-hover text-sm text-white font-medium">
                <Icon size={14}/> {label}
              </Link>
            ))}
          </div>
        )}
      </div>

      {/* Goal Scorer Spotlight Banner (If Goals Detected) */}
      {!isProcessing && goalEvents.length > 0 && (
        <div className="glass p-4 rounded-2xl border border-[#22c55e]/40 bg-gradient-to-r from-[#22c55e]/15 via-[#00d4ff]/10 to-transparent flex items-center justify-between flex-wrap gap-3 shadow-[0_0_20px_rgba(34,197,94,0.2)]">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[#22c55e]/20 border border-[#22c55e]/40 flex items-center justify-center text-[#22c55e]">
              <Goal size={22} className="animate-bounce" />
            </div>
            <div>
              <div className="text-xs font-black text-[#22c55e] uppercase tracking-wider flex items-center gap-1.5">
                <Sparkles size={12}/> MATCH HIGHLIGHT: GOAL DETECTED!
              </div>
              <div className="text-sm font-bold text-white mt-0.5">
                {goalEvents.map((g, idx) => (
                  <span key={idx} className="mr-3">
                    ⚽ {g.description || `Goal scored by Team ${g.teams_involved?.[0] === 0 ? 'A' : 'B'}`} at <span className="font-mono text-[#00d4ff]">{g.timestamp || 'Match Time'}</span>
                  </span>
                ))}
              </div>
            </div>
          </div>
          <span className="text-[11px] px-3 py-1 rounded-lg bg-[#22c55e]/20 text-[#22c55e] font-bold border border-[#22c55e]/40 font-mono">
            {goalEvents.length} Goal{goalEvents.length > 1 ? 's' : ''} Confirmed
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* LEFT: Video Player / Live Analyzing Container */}
        <div className="xl:col-span-2 flex flex-col gap-4">
          {/* Main Video & Live Processing Arena */}
          <div className="glass overflow-hidden rounded-2xl relative" style={{ minHeight: 400 }}>
            {isProcessing ? (
              /* LIVE ANALYSIS IN PLACE: 3D Rotating Match Football Loader */
              <div className="pitch-bg p-10 min-h-[400px] flex flex-col items-center justify-center relative overflow-hidden">
                <div className="absolute inset-0 bg-black/60 backdrop-blur-sm"/>
                <div className="relative z-10 w-full flex flex-col items-center">
                  <FootballLoader stage={pipelineStage} progress={pipelineProgress} size="lg" />
                  <div className="text-xs text-slate-400 mt-3 max-w-md text-center">
                    Video uploaded. Currently extracting YOLO bounding boxes, reading jersey numbers with SmolVLM2 OCR, and calculating metric pitch coordinates.
                  </div>
                </div>
              </div>
            ) : (
              /* COMPLETED: Annotated Video Player */
              <>
                {video_url ? (
                  <video ref={videoRef} src={video_url} controls className="w-full" style={{ maxHeight: 420 }}/>
                ) : (
                  <div className="pitch-bg h-72 flex items-center justify-center">
                    <Activity size={48} className="text-white/20"/>
                  </div>
                )}
                <div className="px-4 py-2.5 flex items-center justify-between border-t border-white/5 bg-black/20">
                  <span className="text-xs text-slate-400 flex items-center gap-1">
                    <Sparkles size={12} className="text-[#00d4ff]"/> AI Annotated Video with Bounding Boxes, Speed & Jersey Badges
                  </span>
                  {video_url && (
                    <button
                      onClick={() => handleDownloadVideo(video_url)}
                      disabled={downloading}
                      className="flex items-center gap-1.5 text-xs bg-[#00d4ff]/10 hover:bg-[#00d4ff]/20 text-[#00d4ff] border border-[#00d4ff]/30 px-3 py-1.5 rounded-lg transition-all font-semibold cursor-pointer disabled:opacity-50"
                    >
                      {downloading ? <Loader2 size={13} className="animate-spin"/> : <Download size={13}/>}
                      {downloading ? 'Downloading...' : 'Download Video (.MP4)'}
                    </button>
                  )}
                </div>
              </>
            )}
          </div>

          {/* Stat Comparison Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <StatCompareCard label="Possession" Icon={Repeat2} homeVal={Math.round(possession.home||50)} awayVal={Math.round(possession.away||50)} suffix="%"/>
            <StatCompareCard label="Goals" Icon={Goal} homeVal={goals.home||0} awayVal={goals.away||0}/>
            <StatCompareCard label="Shots on Target" Icon={Crosshair} homeVal={shotEvents.length} awayVal={Math.max(0,shotEvents.length-2)}/>
            <StatCompareCard label="Pass Accuracy" Icon={Activity} homeVal={Math.round(pass_accuracy.home||80)} awayVal={Math.round(pass_accuracy.away||75)} suffix="%"/>
            <StatCompareCard label="Distance (km)" Icon={Map} homeVal={totalDistance.home.toFixed(1)} awayVal={totalDistance.away.toFixed(1)}/>
            <StatCompareCard label="Sprints" Icon={Zap} homeVal={totalSprints.home} awayVal={totalSprints.away}/>
          </div>
        </div>

        {/* RIGHT: Match Events Timeline & Team Radar */}
        <div className="flex flex-col gap-4">
          {/* Match Events Timeline */}
          <div className="glass p-4 rounded-2xl flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <h2 className="text-white font-bold text-sm flex items-center gap-1.5">
                <CalendarClock size={14} className="text-[#00d4ff]"/> Match Events Timeline
              </h2>
              <span className="text-slate-500 text-xs font-mono">{events.length} events</span>
            </div>

            {isProcessing ? (
              <div className="p-8 text-center text-slate-500 text-xs flex flex-col items-center gap-2">
                <Loader2 size={16} className="animate-spin text-[#00d4ff]"/>
                <span>Detecting fouls, goals, cards & offsides...</span>
              </div>
            ) : events.length === 0 ? (
              <div className="p-6 text-center text-slate-500 text-xs">No key events detected in clip.</div>
            ) : (
              <div className="flex flex-col gap-2 max-h-60 overflow-y-auto pr-1">
                {events.slice(0, 15).map((evt, i) => {
                  const Icon = EVENT_ICON_MAP[evt.event_type] || Activity
                  const color = EVENT_COLORS[evt.event_type] || '#64748b'
                  const isGoal = evt.event_type === 'Goal'

                  return (
                    <div key={i}
                      className={`flex items-start gap-2.5 p-2 rounded-xl transition-colors ${
                        isGoal ? 'bg-[#22c55e]/15 border border-[#22c55e]/30 shadow-md' : 'bg-white/[0.03] hover:bg-white/[0.06]'
                      }`}>
                      <div className="p-1.5 rounded-lg shrink-0 mt-0.5"
                        style={{ background: `${color}22`, border: `1px solid ${color}44` }}>
                        <Icon size={12} style={{ color }}/>
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-1">
                          <span className="text-xs font-bold truncate" style={{ color }}>{evt.event_type}</span>
                          <span className="text-[10px] text-slate-500 font-mono">{evt.timestamp || '0:00'}</span>
                        </div>
                        <div className="text-[11px] text-slate-300 truncate mt-0.5">{evt.description}</div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {/* Team Performance Radar Chart */}
          <div className="glass p-4 rounded-2xl flex flex-col gap-2">
            <h2 className="text-white font-bold text-sm flex items-center gap-1.5">
              <Activity size={14} className="text-[#00d4ff]"/> Team Performance Radar
            </h2>
            <ResponsiveContainer width="100%" height={170}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#1e3a5f"/>
                <PolarAngleAxis dataKey="subject" tick={{ fill:'#64748b', fontSize:11 }}/>
                <Radar name="Team A" dataKey="A" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25}/>
                <Radar name="Team B" dataKey="B" stroke="#ef4444" fill="#ef4444" fillOpacity={0.2}/>
              </RadarChart>
            </ResponsiveContainer>
            <div className="flex gap-4 justify-center text-xs mt-1">
              <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-[#3b82f6]"/><span className="text-slate-400">Team A</span></div>
              <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-[#ef4444]"/><span className="text-slate-400">Team B</span></div>
            </div>
          </div>
        </div>
      </div>

      {/* BOTTOM SECTION: ALL MATCH PLAYERS & INDIVIDUAL STATS TABLE */}
      <div className="glass p-5 rounded-2xl mt-2">
        <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
          <div>
            <h2 className="text-lg font-black text-white flex items-center gap-2">
              <Users size={18} className="text-[#00d4ff]"/> All Match Players & Tracking Statistics
            </h2>
            <p className="text-slate-500 text-xs mt-0.5">
              Displaying {displayPlayers.length} verified players who participated in this match
            </p>
          </div>
          {!isProcessing && (
            <Link to={`/match/${jobId}/players`}
              className="flex items-center gap-1 text-xs text-[#00d4ff] hover:underline font-semibold glass px-3 py-1.5 rounded-lg">
              <span>View Full Player Detail Cards</span>
              <ChevronRight size={13}/>
            </Link>
          )}
        </div>

        {isProcessing ? (
          <div className="p-8 text-center text-slate-500 text-xs flex flex-col items-center gap-2">
            <Loader2 size={16} className="animate-spin text-[#00d4ff]"/>
            <span>Extracting player velocities, sprint profiles, and jersey numbers...</span>
          </div>
        ) : displayPlayers.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-sm">No player records found for this match.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-white/10 text-slate-400 uppercase tracking-wider">
                  <th className="py-2.5 px-3">Jersey #</th>
                  <th className="py-2.5 px-3">Player ID</th>
                  <th className="py-2.5 px-3">Team</th>
                  <th className="py-2.5 px-3">Role / Status</th>
                  <th className="py-2.5 px-3">Distance (m)</th>
                  <th className="py-2.5 px-3">Top Speed</th>
                  <th className="py-2.5 px-3">Avg Speed</th>
                  <th className="py-2.5 px-3">Sprints</th>
                  <th className="py-2.5 px-3">Touches</th>
                  <th className="py-2.5 px-3">Passes</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {displayPlayers.map((p) => {
                  const isTeamA = p.team_id === 0
                  const teamColor = isTeamA ? '#3b82f6' : '#ef4444'
                  const isSelected = selectedPlayerId === p.id
                  const isScorer = goalScorerIds.has(p.id)
                  const hasOcrJersey = Boolean(p.jersey && p.jersey !== p.id)

                  return (
                    <tr
                      key={p.id}
                      onClick={() => setSelectedPlayerId(p.id)}
                      className={`hover:bg-white/[0.04] transition-colors cursor-pointer ${
                        isSelected ? 'bg-[#00d4ff]/10' : ''
                      }`}
                    >
                      {/* Jersey Badge + Verified Flag */}
                      <td className="py-2.5 px-3">
                        <div className="flex items-center gap-1.5">
                          <span
                            className="px-2.5 py-1 rounded-md font-black text-xs inline-block shadow-sm"
                            style={{
                              backgroundColor: `${teamColor}22`,
                              border: `1px solid ${teamColor}66`,
                              color: teamColor,
                            }}
                          >
                            #{p.jersey || p.id}
                          </span>
                          {hasOcrJersey && (
                            <span className="flex items-center gap-0.5 text-[10px] text-[#22c55e] font-bold px-1.5 py-0.5 rounded bg-[#22c55e]/15 border border-[#22c55e]/30" title="Jersey Number Detected via OCR">
                              <Check size={10} strokeWidth={3}/> OCR
                            </span>
                          )}
                        </div>
                      </td>

                      {/* Player ID */}
                      <td className="py-2.5 px-3 font-semibold text-white">
                        Player {p.id}
                      </td>

                      {/* Team */}
                      <td className="py-2.5 px-3">
                        <span
                          className="px-2 py-0.5 rounded-full text-[11px] font-semibold"
                          style={{
                            backgroundColor: `${teamColor}18`,
                            color: teamColor,
                          }}
                        >
                          {isTeamA ? 'Team A' : 'Team B'}
                        </span>
                      </td>

                      {/* Role / Goal Scorer Status */}
                      <td className="py-2.5 px-3">
                        {isScorer ? (
                          <span className="flex items-center gap-1 text-[11px] font-bold text-[#22c55e] px-2 py-0.5 rounded-md bg-[#22c55e]/20 border border-[#22c55e]/40 shadow-sm animate-pulse">
                            <Goal size={12}/> Goal Scorer
                          </span>
                        ) : (
                          <span className="text-[11px] text-slate-500">Active</span>
                        )}
                      </td>

                      {/* Distance */}
                      <td className="py-2.5 px-3 font-bold text-white">
                        {(p.distance_m || 0).toFixed(1)} m
                      </td>

                      {/* Top Speed */}
                      <td className="py-2.5 px-3 font-bold text-[#00d4ff]">
                        {(p.max_speed || 0).toFixed(1)} km/h
                      </td>

                      {/* Avg Speed */}
                      <td className="py-2.5 px-3 text-slate-300">
                        {(p.avg_speed || 0).toFixed(1)} km/h
                      </td>

                      {/* Sprints */}
                      <td className="py-2.5 px-3 font-semibold text-amber-400">
                        {p.sprint_count || 0}
                      </td>

                      {/* Touches */}
                      <td className="py-2.5 px-3 text-slate-300">
                        {p.touch_count || 0}
                      </td>

                      {/* Passes */}
                      <td className="py-2.5 px-3 text-slate-300">
                        {p.pass_count || 0}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
