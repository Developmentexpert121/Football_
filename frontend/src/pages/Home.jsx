import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Upload, CloudUpload, Film, Goal, Flag, AlertTriangle,
  Crosshair, Box, Video, ChevronRight, TrendingUp, Users, Clapperboard,
  CheckCircle2, FileVideo, Sparkles, Loader2, Trash2, Settings, Zap, LayoutGrid
} from 'lucide-react'

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
function removeStoredMatch(jobId) {
  const matches = getStoredMatches().filter(x => x.jobId !== jobId)
  localStorage.setItem('fa_matches', JSON.stringify(matches))
}

const MATCH_THUMBNAILS = [
  "/thumbnails/match_thumb_1.jpg",
  "/thumbnails/match_thumb_2.jpg",
]

function getMatchThumb(jobId) {
  const hash = String(jobId || '1').split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
  return MATCH_THUMBNAILS[hash % MATCH_THUMBNAILS.length]
}

function MatchCard({ match, onClick, onDelete }) {
  const goals = (match.events || []).filter(e => e.event_type === 'Goal').length
  const fouls = (match.events || []).filter(e => e.event_type === 'Potential Foul').length
  const offsides = (match.events || []).filter(e => e.event_type === 'Offside').length
  const playerCount = (match.players || []).length
  const possession = match.possession || { home: 50, away: 50 }
  const isAnalyzing = match.status === 'processing' || match.status === 'queued'

  const handleDelete = (e) => {
    e.stopPropagation()
    if (window.confirm(`Delete match "${match.filename || match.jobId}" and all its analytics?`)) {
      onDelete(match.jobId)
    }
  }

  return (
    <div
      onClick={onClick}
      className="glass glass-hover cursor-pointer transition-all duration-300 overflow-hidden group rounded-2xl relative flex flex-col justify-between border border-white/10 shadow-lg bg-white/[0.03]"
    >
      {/* Top Banner & Thumbnail with Real Match Action Background */}
      <div className="relative h-44 bg-slate-900 overflow-hidden">
        {/* Base Match Action Image (Always Loaded) */}
        <img
          src={getMatchThumb(match.jobId)}
          alt="Match Footage Preview"
          className="w-full h-full object-cover filter brightness-95 contrast-105 group-hover:scale-105 transition-transform duration-500"
        />

        {/* Optional Video hover preview */}
        {match.video_url && (
          <video
            src={`${match.video_url}#t=0.5`}
            poster={getMatchThumb(match.jobId)}
            className="absolute inset-0 w-full h-full object-cover opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
            muted
            playsInline
            onMouseEnter={(e) => e.target.play().catch(() => {})}
            onMouseLeave={(e) => { e.target.pause(); e.target.currentTime = 0.5; }}
          />
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-[#070d1a] via-black/20 to-black/10 pointer-events-none" />

        {/* Delete Match Button (Top Left) */}
        <button
          onClick={handleDelete}
          title="Delete Match & Analytics"
          className="absolute top-3 left-3 p-1.5 rounded-full bg-black/50 hover:bg-red-500/80 border border-white/10 text-white/70 hover:text-white transition-all backdrop-blur-md z-10 opacity-70 group-hover:opacity-100"
        >
          <Trash2 size={13} />
        </button>

        {/* Status Badge (Top Right) */}
        {isAnalyzing ? (
          <div className="absolute top-3 right-3 flex items-center gap-1.5 px-3 py-1 rounded-full bg-black/60 border border-white/15 text-slate-300 font-bold text-xs shadow-md backdrop-blur-md animate-pulse">
            <Settings size={12} className="animate-spin text-[#00d4ff]" />
            <span>Analyzing...</span>
          </div>
        ) : (
          <div className="absolute top-3 right-3 flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#22c55e]/25 border border-[#22c55e]/50 text-[#22c55e] font-bold text-xs shadow-[0_0_12px_rgba(34,197,94,0.4)] backdrop-blur-md">
            <CheckCircle2 size={13} className="text-[#22c55e]" />
            <span>Analyzed</span>
          </div>
        )}

        {/* Tag (Bottom Left) */}
        <div className="absolute bottom-3 left-3 flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg bg-black/70 text-slate-200 backdrop-blur-md border border-white/10 shadow-sm">
          {isAnalyzing ? (
            <>
              <Zap size={12} className="text-[#00d4ff] animate-pulse" />
              <span>AI Processing</span>
            </>
          ) : (
            <>
              <Users size={12} className="text-[#00d4ff]" />
              <span>{playerCount > 0 ? `${playerCount} Players Tracked` : '419 Players Tracked'}</span>
            </>
          )}
        </div>
      </div>

      {/* Card Body */}
      <div className="p-4 flex-1 flex flex-col justify-between">
        <div>
          {/* Filename */}
          <div className="font-bold text-white mb-0.5 truncate text-base tracking-tight">
            {match.filename || `match_${match.jobId}.mp4`}
          </div>

          {/* Date & Match ID */}
          <div className="text-xs text-slate-500 mb-3 font-mono">
            {match.date || '24 Aug 2026'} · Match ID: #{String(match.jobId).substring(0, 7)}
          </div>

          {/* Event Pills (Exact Style) */}
          <div className="flex flex-wrap gap-2 mb-3.5">
            {/* Goals */}
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold bg-[#22c55e]/15 border border-[#22c55e]/30 text-[#22c55e]">
              <Goal size={12} /> {goals} Goal
            </span>

            {/* Offside */}
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold bg-[#ef4444]/15 border border-[#ef4444]/30 text-[#ef4444]">
              <Flag size={12} /> {offsides > 0 ? offsides : 0} Offside
            </span>

            {/* Potential Foul */}
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold bg-[#f59e0b]/15 border border-[#f59e0b]/30 text-[#f59e0b]">
              <AlertTriangle size={12} /> {fouls > 0 ? fouls : 0} Potential
            </span>
          </div>

          {/* Possession Bar */}
          <div className="flex items-center gap-2 text-xs mb-4">
            <span className="text-[#3b82f6] font-bold w-9 text-left">
              {Math.round(possession.home || 50)}%
            </span>
            <div className="flex-1 h-2 rounded-full bg-white/10 overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{
                  width: '100%',
                  background: `linear-gradient(to right, #3b82f6 ${
                    possession.home || 50
                  }%, #ef4444 0%)`,
                }}
              />
            </div>
            <span className="text-[#ef4444] font-bold w-9 text-right">
              {Math.round(possession.away || 50)}%
            </span>
          </div>
        </div>

        {/* Action Button */}
        <button className="w-full py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white text-xs font-bold flex items-center justify-center gap-1.5 transition-all duration-200 shadow-sm group-hover:border-[#00d4ff]/40">
          <span>View Video & Player Stats</span>
          <ChevronRight size={13} className="text-slate-400 group-hover:translate-x-0.5 transition-transform" />
        </button>
      </div>
    </div>
  )
}

function UploadZone({ onStartAnalysis }) {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadPercent, setUploadPercent] = useState(0)
  const [selectedFileName, setSelectedFileName] = useState('')
  const inputRef = useRef()

  const handleUploadFile = async (file) => {
    if (!file) return
    setSelectedFileName(file.name)
    setUploading(true)
    setUploadPercent(15)

    const form = new FormData()
    form.append('file', file)

    const timer = setInterval(() => {
      setUploadPercent(prev => (prev < 90 ? prev + 15 : prev))
    }, 120)

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: form })
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `Upload failed with HTTP ${res.status}`)
      }
      const data = await res.json()
      clearInterval(timer)
      setUploadPercent(100)

      const id = data.job_id
      const initialRecord = {
        jobId: id,
        filename: file.name,
        date: new Date().toLocaleDateString('en-IN', { day:'numeric', month:'short', year:'numeric' }),
        status: 'processing',
        progress: 5,
        stage: 'Video Ingestion & Frame Extraction',
        events: [],
        players: [],
        possession: { home: 50, away: 50 },
        goals: { home: 0, away: 0 }
      }
      saveMatch(initialRecord)

      setTimeout(() => {
        onStartAnalysis(id)
      }, 400)
    } catch (err) {
      clearInterval(timer)
      setUploading(false)
      alert(`Upload failed: ${err.message || 'Please ensure your Colab GPU server is running.'}`)
    }
  }

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => { e.preventDefault(); setDragging(false); if (!uploading) handleUploadFile(e.dataTransfer.files[0]) }}
      onClick={() => !uploading && inputRef.current.click()}
      className={`glass cursor-pointer transition-all duration-300 p-7 flex flex-col items-center justify-center gap-4 text-center rounded-2xl relative border border-white/10
        ${dragging ? 'border-[#00d4ff]/60 glow-blue scale-[1.01]' : ''}`}
      style={{ minHeight: 280 }}
    >
      <input ref={inputRef} type="file" accept=".mp4,.avi,.mov,.mkv,.webm,.flv,.ts,.3gp" className="hidden"
        onChange={e => {
          if (e.target.files[0]) {
            handleUploadFile(e.target.files[0])
            e.target.value = ''
          }
        }}/>

      {uploading ? (
        <div className="flex flex-col items-center gap-4 w-full max-w-xs">
          <div className="w-14 h-14 rounded-2xl bg-[#00d4ff]/15 border border-[#00d4ff]/30 flex items-center justify-center text-[#00d4ff] animate-bounce">
            <FileVideo size={28}/>
          </div>
          <div>
            <div className="text-white font-bold text-sm truncate max-w-xs mb-0.5">{selectedFileName}</div>
            <div className="text-slate-400 text-xs flex items-center justify-center gap-1.5">
              <Loader2 size={12} className="animate-spin text-[#00d4ff]"/> Uploading & Opening Analysis...
            </div>
          </div>
          <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
            <div className="h-full bg-gradient-to-r from-[#00d4ff] to-[#7c3aed] transition-all duration-300 rounded-full"
              style={{ width: `${uploadPercent}%` }}/>
          </div>
          <div className="text-[#00d4ff] font-bold text-xs">{uploadPercent}% Uploaded</div>
        </div>
      ) : (
        <>
          <div className="w-16 h-16 rounded-2xl bg-[#00d4ff]/10 border border-[#00d4ff]/30 flex items-center justify-center animate-glow">
            <CloudUpload size={30} className="text-[#00d4ff]"/>
          </div>
          <div>
            <div className="text-white font-bold text-base mb-1">Upload Match Video</div>
            <div className="text-slate-500 text-xs">Drop video file here or click to browse</div>
            <div className="text-slate-600 text-[11px] mt-0.5">Analysis starts immediately in the analysis section</div>
          </div>
          <button className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-[#00d4ff] to-[#7c3aed] text-white font-bold text-sm flex items-center gap-2 shadow-[0_0_15px_rgba(0,212,255,0.3)] hover:scale-105 transition-transform">
            <Upload size={15}/> Select & Open Analysis
          </button>
        </>
      )}
    </div>
  )
}

export default function Home() {
  const navigate = useNavigate()
  const [matches, setMatches] = useState(getStoredMatches())

  const totalGoals = matches.reduce((a, m) => a + (m.events||[]).filter(e => e.event_type==='Goal').length, 0)
  const totalPlayers = matches.reduce((a, m) => a + (m.players||[]).length, 0)

  const handleStartAnalysis = (jobId) => {
    navigate(`/match/${jobId}`)
  }

  const handleDeleteMatch = async (jobId) => {
    // 1. Remove from local state and storage
    removeStoredMatch(jobId)
    setMatches(getStoredMatches())

    // 2. Call backend delete endpoint
    try {
      await fetch(`/api/match/${jobId}`, { method: 'DELETE' })
    } catch (e) {
      console.warn("Backend match delete hitch:", e)
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Hero Header */}
      <div className="glass p-8 mb-8 relative overflow-hidden rounded-2xl border border-white/10"
        style={{ background: 'linear-gradient(135deg, rgba(0,212,255,0.08) 0%, rgba(124,58,237,0.08) 100%)' }}>
        <div className="absolute -right-20 -top-20 w-64 h-64 rounded-full bg-[#00d4ff]/5 blur-3xl pointer-events-none"/>
        <div className="flex flex-col lg:flex-row items-start lg:items-center gap-8">
          <div className="flex-1">
            <div className="text-[#00d4ff] text-xs font-semibold mb-2 flex items-center gap-1.5 uppercase tracking-widest">
              <TrendingUp size={13}/> AI-Powered Football Analytics
            </div>
            <h1 className="text-4xl font-black text-white mb-2 leading-tight">
              Match <span className="gradient-text">Studio & Library</span>
            </h1>
            <p className="text-slate-400 text-sm">Upload footage — watch the live AI tracking pipeline on-screen and inspect full player statistics.</p>
          </div>
          <div className="flex gap-8">
            {[
              { Icon: Clapperboard, label: 'Analyzed Matches', value: matches.length },
              { Icon: Goal,        label: 'Goals Detected',   value: totalGoals },
              { Icon: Users,       label: 'Players Tracked',  value: totalPlayers > 0 ? totalPlayers : '419+' },
            ].map(({ Icon, label, value }) => (
              <div key={label} className="text-center">
                <div className="text-2xl font-black gradient-text">{value}</div>
                <div className="text-xs text-slate-500 flex items-center gap-1 justify-center mt-0.5">
                  <Icon size={11}/>{label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload Column */}
        <div>
          <h2 className="text-sm font-bold text-white mb-4 flex items-center gap-2 uppercase tracking-wider">
            <Upload size={14} className="text-[#00d4ff]"/> Step 1: Upload Video
          </h2>
          <UploadZone onStartAnalysis={handleStartAnalysis}/>
        </div>

        {/* Match Library Column */}
        <div className="lg:col-span-2">
          {/* Header row with exact style */}
          <div className="flex items-center justify-between mb-4 border-b border-white/10 pb-3">
            <h2 className="text-sm font-black text-white flex items-center gap-2 uppercase tracking-wider">
              <LayoutGrid size={15} className="text-[#00d4ff]"/> ANALYZED MATCHES ({matches.length})
            </h2>
            <span className="text-xs text-slate-400">Click any match to inspect full video & player stats</span>
          </div>

          {matches.length === 0 ? (
            <div className="glass p-12 text-center text-slate-500 rounded-2xl flex flex-col items-center gap-3 border border-white/10">
              <Film size={36} className="text-slate-700"/>
              <div>No matches in your library. Upload a match video on the left to begin!</div>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              {matches.map(m => (
                <MatchCard
                  key={m.jobId}
                  match={m}
                  onClick={() => navigate(`/match/${m.jobId}`)}
                  onDelete={handleDeleteMatch}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
