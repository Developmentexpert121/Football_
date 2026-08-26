import { Link, useLocation } from 'react-router-dom'
import { Activity, LayoutDashboard, Users, CalendarClock, Map, Sun, Moon } from 'lucide-react'
import { useTheme } from '../context/ThemeContext'

export default function Navbar() {
  const { pathname } = useLocation()
  const { theme, toggleTheme } = useTheme()
  const jobId = pathname.split('/')[2]

  const navLinks = jobId ? [
    { to: `/match/${jobId}`, icon: <Activity size={16}/>, label: 'Analysis' },
    { to: `/match/${jobId}/players`, icon: <Users size={16}/>, label: 'Players' },
    { to: `/match/${jobId}/events`, icon: <CalendarClock size={16}/>, label: 'Events' },
    { to: `/match/${jobId}/tactics`, icon: <Map size={16}/>, label: 'Tactics' },
  ] : []

  return (
    <nav
      className="sticky top-0 z-50 glass border-b px-6 py-3 flex items-center justify-between transition-all duration-300"
      style={{ borderRadius: 0 }}
    >
      {/* Logo */}
      <Link to="/" className="flex items-center gap-2.5 group">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#00d4ff] to-[#7c3aed] flex items-center justify-center shadow-[0_0_15px_rgba(0,212,255,0.4)] group-hover:scale-105 transition-transform">
          <span className="text-white font-black text-sm">⚽</span>
        </div>
        <span className="font-black text-lg tracking-tight gradient-text">FootballAI</span>
      </Link>

      {/* Dynamic nav for match pages */}
      {navLinks.length > 0 && (
        <div className="flex items-center gap-1.5">
          {navLinks.map(({ to, icon, label }) => (
            <Link
              key={to}
              to={to}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200
                ${
                  pathname === to
                    ? 'bg-[#00d4ff]/15 text-[#00d4ff] border border-[#00d4ff]/30 shadow-[0_0_12px_rgba(0,212,255,0.15)] font-semibold'
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`}
            >
              {icon}
              {label}
            </Link>
          ))}
        </div>
      )}

      {/* Right controls: Theme toggle & Library */}
      <div className="flex items-center gap-3">
        {/* Dark/Light Mode Switch Button */}
        <button
          onClick={toggleTheme}
          title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
          className="flex items-center gap-2 px-3 py-1.5 rounded-xl border border-white/10 glass glass-hover text-xs font-semibold text-slate-300 hover:text-white transition-all cursor-pointer shadow-sm"
        >
          {theme === 'dark' ? (
            <>
              <Sun size={15} className="text-amber-400 animate-[spin_10s_linear_infinite]" />
              <span className="hidden sm:inline">Light Mode</span>
            </>
          ) : (
            <>
              <Moon size={15} className="text-indigo-400" />
              <span className="hidden sm:inline">Dark Mode</span>
            </>
          )}
        </button>

        {/* Home icon */}
        <Link
          to="/"
          className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-sm text-slate-400 hover:text-white hover:bg-white/5 glass transition-all"
        >
          <LayoutDashboard size={15} /> <span className="hidden sm:inline">Library</span>
        </Link>
      </div>
    </nav>
  )
}
