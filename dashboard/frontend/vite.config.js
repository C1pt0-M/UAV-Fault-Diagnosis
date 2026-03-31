import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const backendTarget = process.env.VITE_BACKEND_TARGET || 'http://localhost:5000'

export default defineConfig({
  plugins: [vue()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined

          if (id.includes('vue-router')) return 'vue-router'
          return undefined
        }
      }
    }
  },
  test: {
    environment: 'jsdom'
  },
  server: {
    port: 5173,
    open: false,
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
        secure: false
      },
      '/health': {
        target: backendTarget,
        changeOrigin: true,
        secure: false
      }
    }
  },
  resolve: {
    alias: {
      '@': '/src'
    }
  }
})
