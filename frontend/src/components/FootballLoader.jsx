export default function FootballLoader({ stage, progress, size = "md" }) {
  const sizeMap = {
    sm: { ball: "w-16 h-16", glow: "w-20 h-20", ring: "w-24 h-24" },
    md: { ball: "w-24 h-24", glow: "w-28 h-28", ring: "w-36 h-36" },
    lg: { ball: "w-32 h-32", glow: "w-36 h-36", ring: "w-44 h-44" },
  }

  const s = sizeMap[size] || sizeMap.md

  return (
    <div className="flex flex-col items-center justify-center gap-5 my-2">
      {/* 3D Ball Container with Orbitals */}
      <div className="relative flex items-center justify-center">
        {/* Outer Orbital Pulse Ring */}
        <div
          className={`absolute ${s.ring} rounded-full border border-[#00d4ff]/40 animate-ping opacity-25 pointer-events-none`}
        />
        <div
          className={`absolute ${s.ring} rounded-full border border-dashed border-[#00d4ff]/30 animate-spin pointer-events-none`}
          style={{ animationDuration: "12s" }}
        />

        {/* Ambient Neon Glow */}
        <div
          className={`absolute ${s.glow} rounded-full bg-gradient-to-r from-[#00d4ff]/40 to-[#7c3aed]/40 blur-xl pointer-events-none animate-pulse`}
        />

        {/* 3D Match Football Image with 3D Rotation Animation */}
        <div className={`relative ${s.ball} rounded-full overflow-hidden shadow-[0_0_30px_rgba(0,212,255,0.45)] transition-transform`}>
          <img
            src="/classic_bw_football.jpg"
            alt="Classic Black & White Football"
            className="w-full h-full object-cover rounded-full animate-[spin_4s_linear_infinite]"
          />
          {/* 3D Sphere Specular Highlight Overlay */}
          <div className="absolute inset-0 rounded-full bg-gradient-to-tr from-transparent via-white/10 to-white/30 pointer-events-none" />
          <div className="absolute inset-0 rounded-full shadow-[inset_-6px_-6px_14px_rgba(0,0,0,0.8),inset_6px_6px_14px_rgba(0,212,255,0.4)] pointer-events-none" />
        </div>
      </div>

      {/* Stage and Progress Info */}
      <div className="flex flex-col items-center gap-2 max-w-sm w-full text-center">
        <div className="text-white font-bold text-base tracking-wide flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-[#00d4ff] animate-ping" />
          {stage || "Analyzing Match..."}
        </div>

        {/* Glowing Progress Bar */}
        <div className="w-full bg-white/10 rounded-full h-2.5 overflow-hidden p-0.5 border border-white/5 shadow-inner">
          <div
            className="h-full rounded-full bg-gradient-to-r from-[#00d4ff] via-[#38bdf8] to-[#818cf8] transition-all duration-500 shadow-[0_0_12px_#00d4ff]"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Percentage */}
        <div className="text-2xl font-black gradient-text tracking-tight mt-0.5">
          {progress}%
        </div>
      </div>
    </div>
  )
}
