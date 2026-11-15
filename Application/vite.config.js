import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // the PlantCare AI backend (Application/backend) runs on :8000
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/files': 'http://localhost:8000',
    },
  },
  preview: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/files': 'http://localhost:8000',
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom', 'react-router-dom'],
          motion: ['framer-motion'],
          charts: ['recharts'],
        },
      },
    },
  },
})
