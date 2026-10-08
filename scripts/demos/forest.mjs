// The jungle demo: CC0 trees, shrubs and forest-floor models from Poly Haven (millions of triangles each), mixed
// with procedural palms, banana plants and ferns generated at load time. Downloads them (once) and opens the scene:
//
//   npm run demo:forest                 # trees and forest floor (about 100 MB of downloads)
//   npm run demo:forest -- --no-open    # prepare only, don't start a server
//   npm run demo:forest -- --dev        # open it on the dev server (live reload) instead of the production build
//
// Data lands in examples/scans/forest/ (not committed), served at /scans/forest/ by vite.config.ts. Re-running skips
// everything that is already there and resumes interrupted downloads.
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const out = join(root, 'examples', 'scans', 'forest');
const args = process.argv.slice(2);

// Poly Haven asset ids (all CC0), by role in the scene.
// Broadleaf trees and shrubs with modelled leaves.
// Shrubs and jungle-floor plants (the trees, palms and bananas are generated in code).
const TREES = ['calathea_orbifolia_01', 'shrub_01', 'shrub_02', 'shrub_03', 'shrub_04', 'nettle_plant'];
const GROUND = ['grass_medium_01', 'celandine_01', 'moss_01', 'bark_debris_01', 'root_cluster_02', 'rock_moss_set_01'];
const assets = [...TREES, ...GROUND];
/** Photoscanned forest-floor material for the terrain (2 x 2 m per tile). */
const GROUND_TEXTURE = 'forrest_ground_01';

async function json(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.json();
}

/**
 * Downloads to `file` via `file.part`, so an interrupted download is never mistaken for a complete one. Uses curl
 * (part of macOS, Linux and Windows 10+), which resumes dropped transfers with range requests.
 */
async function download(url, file, size) {
  if (existsSync(file) && (!size || statSync(file).size === size)) return;
  mkdirSync(dirname(file), { recursive: true });
  const part = `${file}.part`;
  for (let attempt = 1; ; attempt++) {
    const code = await new Promise((resolve) => {
      const curl = spawn('curl', ['--location', '--fail', '--silent', '--show-error', '--continue-at', '-', '--retry', '10', '--retry-all-errors', '--retry-delay', '3', '--output', part, url], { stdio: ['ignore', 'ignore', 'inherit'] });
      curl.on('close', resolve);
      curl.on('error', () => resolve(-1));
    });
    const have = existsSync(part) ? statSync(part).size : 0;
    if (code === 0 && (!size || have === size)) break;
    if (code === -1) throw new Error('curl is required to download the forest models');
    if (attempt >= 20) throw new Error(`download failed: ${url} (curl exit ${code}, ${have} of ${size} bytes)`);
  }
  renameSync(part, file);
}

/** Runs `tasks` with at most `limit` in flight. */
async function pool(tasks, limit) {
  let next = 0;
  await Promise.all(Array.from({ length: limit }, async () => {
    while (next < tasks.length) await tasks[next++]();
  }));
}

const credits = [];
const tasks = [];
for (const id of assets) {
  const [info, files] = await Promise.all([json(`https://api.polyhaven.com/info/${id}`), json(`https://api.polyhaven.com/files/${id}`)]);
  const gltf = files.gltf['2k'].gltf;
  const dir = join(out, id);
  tasks.push(() => download(gltf.url, join(dir, `${id}.gltf`), gltf.size));
  for (const [path, file] of Object.entries(gltf.include)) tasks.push(() => download(file.url, join(dir, path), file.size));
  credits.push({
    id,
    name: info.name,
    role: TREES.includes(id) ? 'tree' : 'ground',
    file: `${id}/${id}.gltf`,
    triangles: info.polycount,
    source: `https://polyhaven.com/a/${id}`,
    authors: Object.keys(info.authors ?? {}),
    license: 'CC0 1.0',
  });
}
{
  const [info, files] = await Promise.all([json(`https://api.polyhaven.com/info/${GROUND_TEXTURE}`), json(`https://api.polyhaven.com/files/${GROUND_TEXTURE}`)]);
  for (const map of ['Diffuse', 'nor_gl', 'arm']) {
    const file = files[map]['2k'].jpg;
    tasks.push(() => download(file.url, join(out, 'textures', GROUND_TEXTURE, file.url.split('/').pop()), file.size));
  }
  credits.push({
    id: GROUND_TEXTURE,
    name: info.name,
    role: 'terrain',
    file: `textures/${GROUND_TEXTURE}/`,
    triangles: 0,
    source: `https://polyhaven.com/a/${GROUND_TEXTURE}`,
    authors: Object.keys(info.authors ?? {}),
    license: 'CC0 1.0',
  });
}
mkdirSync(out, { recursive: true });
writeFileSync(join(out, 'credits.json'), JSON.stringify(credits, null, 2) + '\n');

console.log(`Downloading ${tasks.length} files into ${out} (skipping complete ones) ...`);
let done = 0;
await pool(tasks.map((task) => async () => {
  await task();
  if (++done % 10 === 0 || done === tasks.length) console.log(`  ${done}/${tasks.length}`);
}), 4);
console.log(`Downloaded: ${credits.length} models. Credits in scans/forest/credits.json`);

if (args.includes('--dev')) {
  console.log('Starting the dev server: http://localhost:8090/?scene=forest');
  spawn('npx', ['vite', '--open', '/?scene=forest'], { cwd: root, stdio: 'inherit', shell: process.platform === 'win32' });
} else if (!args.includes('--no-open')) {
  // The production build, served statically: what users get, and noticeably faster than the dev server.
  console.log('Building the demo ...');
  const built = spawnSync('npx', ['vite', 'build', '--logLevel', 'warn'], { cwd: root, stdio: 'inherit', shell: process.platform === 'win32' });
  if (built.status !== 0) throw new Error('building the demo failed');
  console.log('Serving it: http://localhost:8092/?scene=forest&tour');
  spawn('npx', ['vite', 'preview', '--open', '/?scene=forest&tour'], { cwd: root, stdio: 'inherit', shell: process.platform === 'win32' });
}
