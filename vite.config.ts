import { defineConfig } from 'vite';

// The demo app (examples/). `npm run dev` serves it, `npm run build:demo` builds it for GitHub Pages.
export default defineConfig({
  root: 'examples',
  base: './',
  server: { port: 8090 },
  build: { outDir: '../dist-demo', emptyOutDir: true, target: 'es2022' },
});
