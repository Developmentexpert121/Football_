import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { ThemeProvider } from './context/ThemeContext'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import MatchAnalysis from './pages/MatchAnalysis'
import Players from './pages/Players'
import Events from './pages/Events'
import Tactics from './pages/Tactics'

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <div className="min-h-screen flex flex-col transition-colors duration-300">
          <Navbar />
          <main className="flex-1">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/match/:jobId" element={<MatchAnalysis />} />
              <Route path="/match/:jobId/players" element={<Players />} />
              <Route path="/match/:jobId/events" element={<Events />} />
              <Route path="/match/:jobId/tactics" element={<Tactics />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </ThemeProvider>
  )
}
