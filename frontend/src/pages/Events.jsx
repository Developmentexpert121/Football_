import { useParams } from 'react-router-dom'
import { useEffect, useState } from 'react'
import {
  Goal, Flag, AlertTriangle, Crosshair, Box, Activity, RefreshCw, ShieldOff, CornerUpRight
} from 'lucide-react'
import FootballLoader from '../components/FootballLoader'

function getStoredMatch(jobId) {
  try {
    return JSON.parse(localStorage.getItem('fa_matches') || '[]').find(m => m.jobId === jobId) || null
  } catch { return null }
}

const EVENT_ICON_MAP = {
  Goal: Goal,
  'Potential Foul': AlertTriangle,
  Offside: ShieldOff,
  'Shot on Target': Crosshair,
  'Penalty Area Entry': Box,
  'Yellow Card Candidate': AlertTriangle,
  'Corner Kick': CornerUpRight,
  'Free Kick': Activity,
}
const EVENT_COLORS = {
  Goal: '#22c55e', 'Potential Foul': '#f59e0b', Offside: '#ef4444',
  'Shot on Target': '#a855f7', 'Penalty Area Entry': '#3b82f6',
  'Yellow Card Candidate': '#eab308', 'Corner Kick': '#06b6d4', 'Free Kick': '#84cc16',
}
const ALL_TYPES = ['Goal', 'Potential Foul', 'Offside', 'Shot on Target', 'Penalty Area Entry', 'Corner Kick']

export default function Events() {
  const { jobId } = useParams()
  const [events, setEvents] = useState([])
  const [filter, setFilter] = useState('All')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const stored = getStoredMatch(jobId)
    if (stored?.events) { setEvents(stored.events); setLoading(false); return }
    fetch(`/api/results/${jobId}`).then(r => r.json()).then(d => { setEvents(d.events||[]); setLoading(false) })
  }, [jobId])

  const filtered = filter === 'All' ? events : events.filter(e => e.event_type === filter)
  const counts = {}
  events.forEach(e => { counts[e.event_type] = (counts[e.event_type]||0)+1 })

  const maxSeconds = events.reduce((m, e) => {
    const parts = (e.timestamp||'0:00').split(':').map(Number)
    return Math.max(m, parts[0]*60+(parts[1]||0))
  }, 1)

  return (
    <div className="max-w-5xl mx-auto px-4 py-6">
      <div className="mb-6">
        <h1 className="text-2xl font-black gradient-text">Events Timeline</h1>
        <p className="text-slate-500 text-xs mt-0.5">{events.length} events detected across this match</p>
      </div>

      {/* Filter pills */}
      <div className="flex flex-wrap gap-2 mb-6">
        <button onClick={() => setFilter('All')}
          className={`px-4 py-1.5 rounded-full text-sm font-semibold transition-all border
            ${filter==='All' ? 'bg-white/15 text-white border-white/30' : 'glass text-slate-400 border-white/10 hover:text-white'}`}>
          All ({events.length})
        </button>
        {ALL_TYPES.filter(t => counts[t]).map(t => {
          const Icon = EVENT_ICON_MAP[t] || Activity
          const color = EVENT_COLORS[t]
          return (
            <button key={t} onClick={() => setFilter(t)}
              className="px-3 py-1.5 rounded-full text-sm font-semibold transition-all border flex items-center gap-1.5"
              style={{
                background: filter===t ? `${color}25` : 'rgba(255,255,255,0.04)',
                borderColor: filter===t ? color : 'rgba(255,255,255,0.08)',
                color: filter===t ? color : '#94a3b8',
              }}>
              <Icon size={12} strokeWidth={2}/>{t.split(' ')[0]} ({counts[t]})
            </button>
          )
        })}
      </div>

      {/* Timeline scrubber */}
      <div className="glass p-4 mb-6 rounded-2xl">
        <div className="text-xs text-slate-500 mb-3 font-medium uppercase tracking-wider">Match Timeline</div>
        <div className="relative h-5">
          <div className="absolute inset-y-0 left-0 right-0 flex items-center">
            <div className="w-full h-0.5 bg-white/10 rounded-full relative">
              {events.map((e, i) => {
                const parts = (e.timestamp||'0:00').split(':').map(Number)
                const secs = parts[0]*60+(parts[1]||0)
                const pct = (secs / Math.max(maxSeconds,1)) * 100
                const Icon = EVENT_ICON_MAP[e.event_type]
                return (
                  <div key={i} title={`${e.event_type} @ ${e.timestamp}`}
                    className="absolute w-3 h-3 rounded-full -translate-y-1/2 top-1/2 -translate-x-1/2 border-2 border-[#070d1a]
                               hover:scale-150 transition-transform cursor-pointer z-10 flex items-center justify-center"
                    style={{ left:`${pct}%`, backgroundColor: EVENT_COLORS[e.event_type]||'#64748b' }}/>
                )
              })}
            </div>
          </div>
        </div>
        <div className="flex justify-between text-xs text-slate-700 mt-2 font-mono">
          <span>0:00</span><span>45:00</span><span>90:00</span>
        </div>
      </div>

      {/* Event cards */}
      {loading ? (
        <div className="flex items-center justify-center py-16">
          <FootballLoader stage="Loading Events..." progress={100} size="md" />
        </div>
      ) : filtered.length === 0 ? (
        <div className="glass p-12 text-center text-slate-500 rounded-2xl flex flex-col items-center gap-3">
          <Activity size={36} className="text-slate-700"/>
          <div>No events found for this filter.</div>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {filtered.map((e, i) => {
            const Icon = EVENT_ICON_MAP[e.event_type] || Activity
            const color = EVENT_COLORS[e.event_type] || '#64748b'
            return (
              <div key={i} className="glass glass-hover flex items-start gap-4 p-4 rounded-2xl transition-all duration-200"
                style={{ borderLeft: `4px solid ${color}` }}>
                <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
                  style={{ background: `${color}20` }}>
                  <Icon size={16} strokeWidth={2} style={{ color }}/>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 flex-wrap mb-1">
                    <span className="font-bold text-sm" style={{ color }}>{e.event_type}</span>
                    <span className="text-xs text-slate-500 font-mono">{e.timestamp}</span>
                    {e.jerseys_involved?.length > 0 && (
                      <span className="text-xs text-[#00d4ff] bg-[#00d4ff]/10 border border-[#00d4ff]/20 px-2 py-0.5 rounded-full">
                        #{e.jerseys_involved.join(', #')}
                      </span>
                    )}
                    {e.is_replay_skipped && (
                      <span className="text-xs text-slate-500 bg-white/5 border border-white/10 px-2 py-0.5 rounded-full flex items-center gap-1">
                        <RefreshCw size={9}/> Replay Skipped
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-slate-400">{e.description}</p>
                  <div className="flex items-center gap-2 mt-2">
                    <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width:`${(e.confidence||0)*100}%`, backgroundColor: color }}/>
                    </div>
                    <span className="text-xs text-slate-500">{Math.round((e.confidence||0)*100)}% confidence</span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
