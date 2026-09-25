import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import dts from 'vite-plugin-dts'

// Build a standalone library: `import { WinButton } from 'winui-web-library'`.
// `vite` (serve) still uses the project `index.html` for the demo app in /dev.
export default defineConfig({
  plugins: [
    vue(),
    // Emit `dist/*.d.ts` so consumers get typed named imports.
    dts({ tsconfigPath: './tsconfig.app.json', include: ['src'] }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    lib: {
      entry: fileURLToPath(new URL('./src/index.ts', import.meta.url)),
      name: 'WinUIWebLibrary',
      formats: ['es', 'cjs'],
      fileName: (format) => (format === 'es' ? 'index.mjs' : 'index.cjs'),
    },
    cssCodeSplit: false,
    rollupOptions: {
      // vue stays a peer dependency of the consumer app
      external: ['vue'],
      output: {
        globals: { vue: 'Vue' },
        assetFileNames: 'winui-web-library.css',
      },
    },
  },
})