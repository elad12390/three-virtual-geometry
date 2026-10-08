import { defineConfig } from 'vite';

const banner =
  '/*! three-virtual-geometry | MIT License | Independent implementation, not affiliated with Epic Games. ' +
  'Includes code derived from nanite-webgpu (MIT, Copyright (c) 2024 Marcin Matuszczyk). See THIRD_PARTY_NOTICES.md */';

// The library: dist/index.js (ES module). three and meshoptimizer stay external (peer / dependency).
export default defineConfig({
  publicDir: false,
  build: {
    lib: { entry: 'src/index.ts', formats: ['es'], fileName: 'index' },
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2022',
    sourcemap: true,
    rollupOptions: {
      external: [/^three(\/.*)?$/, 'meshoptimizer'],
      output: { banner },
    },
  },
});
