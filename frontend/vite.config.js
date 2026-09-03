import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'https://identifies-records-dam-farming.trycloudflare.com',
        changeOrigin: true,
        secure: false,
        timeout: 600000,
        proxyTimeout: 600000,
        headers: {
          'ngrok-skip-browser-warning': 'true'
        },
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, res) => {
            console.log('[Vite Proxy Warning] Connection reset by backend tunnel. Please check your Colab server status.');
            if (res && !res.headersSent) {
              res.writeHead(502, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ error: 'Backend GPU tunnel disconnected or restarted.' }));
            }
          });
        }
      },
      '/media': {
        target: 'https://identifies-records-dam-farming.trycloudflare.com',
        changeOrigin: true,
        secure: false,
        timeout: 600000,
        proxyTimeout: 600000,
        headers: {
          'ngrok-skip-browser-warning': 'true'
        }
      },
    }
  }
})
