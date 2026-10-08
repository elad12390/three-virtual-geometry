import { createReadStream, existsSync, statSync } from 'node:fs';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Connect, type Plugin } from 'vite';

// Downloaded scan data for `?scene=scans` (npm run demo:scans). It lives outside public/ so production builds don't
// copy gigabytes into dist-demo; this plugin serves it at /scans/ from both the dev server and `vite preview`.
const SCANS = fileURLToPath(new URL('./examples/scans', import.meta.url));
const TYPES: Record<string, string> = {
  '.json': 'application/json',
  '.gltf': 'model/gltf+json',
  '.bin': 'application/octet-stream',
  '.jpg': 'image/jpeg',
  '.png': 'image/png',
  '.vgeo': 'application/octet-stream',
  '.stl': 'model/stl',
};

function serveScans(): Plugin {
  const middleware: Connect.NextHandleFunction = (req, res, next) => {
    const url = decodeURIComponent((req.url ?? '').split('?')[0]);
    const prefix = url.indexOf('/scans/');
    if (prefix === -1) return next();
    const file = normalize(join(SCANS, url.slice(prefix + '/scans/'.length)));
    if (!file.startsWith(SCANS) || !existsSync(file) || !statSync(file).isFile()) {
      res.statusCode = 404;
      res.end('not found');
      return;
    }
    res.setHeader('Content-Type', TYPES[extname(file).toLowerCase()] ?? 'application/octet-stream');
    res.setHeader('Content-Length', statSync(file).size);
    createReadStream(file).pipe(res);
  };
  return {
    name: 'serve-scans',
    configureServer: (server) => void server.middlewares.use(middleware),
    configurePreviewServer: (server) => void server.middlewares.use(middleware),
  };
}

// The demo app (examples/). `npm run dev` serves it, `npm run build:demo` builds it for GitHub Pages.
export default defineConfig({
  root: 'examples',
  base: './',
  server: { port: 8090 },
  preview: { port: 8092 },
  build: { outDir: '../dist-demo', emptyOutDir: true, target: 'es2022' },
  plugins: [serveScans()],
});
