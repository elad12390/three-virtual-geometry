// The scanned-environment demo: real photogrammetry and 3D scans, the kind of data virtual geometry was made for.
// Downloads the scans (once), bakes the statues, and opens the demo:
//
//   npm run demo:scans                 # everything: about 2 GB of downloads, 13 rock scans and 9 statues
//   npm run demo:scans -- --small      # a quick subset (3 rocks, 2 statues)
//   npm run demo:scans -- --no-open    # prepare only, don't start a server
//   npm run demo:scans -- --dev        # open it on the dev server (live reload) instead of the production build
//
// Data lands in examples/scans/ (not committed), served at /scans/ by vite.config.ts. Re-running skips everything that is already there.
//
// Sources, all public domain / CC0:
//   - Poly Haven (https://polyhaven.com, CC0): photoscanned coastal cliffs and rocks, glTF with 2K textures.
//   - SMK, National Gallery of Denmark (https://open.smk.dk, public domain): 3D scans of plaster casts of classical
//     sculpture, STL.
// Files that are already complete are skipped, so the script can be re-run to resume.
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const out = join(root, 'examples', 'scans');
const small = process.argv.includes('--small');

const POLY_HAVEN = small
  ? ['coastal_cliff_04', 'coast_rocks_02', 'coast_land_rocks_03']
  : [
      'coastal_cliff_01', 'coastal_cliff_02', 'coastal_cliff_04', 'coast_land_rocks_02', 'coast_land_rocks_03',
      'coast_land_rocks_04', 'coast_rocks_01', 'coast_rocks_02', 'coast_rocks_03', 'coast_rocks_05', 'coast_line_01',
      'coast_line_02', 'sand_rocks_small_01',
    ];
// SMK object numbers: public-domain sculpture scans, 1 to 4 million triangles each (largest first, then iconic smaller ones).
const SMK = small
  ? ['KAS1312', 'KAS1026']
  : ['KAS35', 'KAS224', 'KAS1312', 'KAS2051', 'KAS1026', 'KAS357', 'KAS403', 'KAS255', 'KAS402', 'KAS248', 'KAS633', 'KAS629', 'KAS196', 'KAS79'];

async function json(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.json();
}

/**
 * Downloads to `file` via `file.part`, so an interrupted download is never mistaken for a complete one. Uses curl
 * (part of macOS, Linux and Windows 10+): some servers drop long transfers, and curl resumes them with range requests.
 */
async function download(url, file, size) {
  if (existsSync(file) && (!size || statSync(file).size === size)) return false;
  mkdirSync(dirname(file), { recursive: true });
  const part = `${file}.part`;
  for (let attempt = 1; ; attempt++) {
    const code = await new Promise((resolve) => {
      const curl = spawn('curl', ['--location', '--fail', '--silent', '--show-error', '--continue-at', '-', '--retry', '10', '--retry-all-errors', '--retry-delay', '5', '--output', part, url], { stdio: ['ignore', 'ignore', 'inherit'] });
      curl.on('close', resolve);
      curl.on('error', () => resolve(-1));
    });
    const have = existsSync(part) ? statSync(part).size : 0;
    if (code === 0 && (!size || have === size)) break;
    if (code === -1) throw new Error('curl is required to download the scans');
    if (attempt >= 20) throw new Error(`download failed: ${url} (curl exit ${code}, ${have} of ${size} bytes)`);
  }
  renameSync(part, file);
  return true;
}

/** Runs `tasks` with at most `limit` in flight. */
async function pool(tasks, limit) {
  let next = 0;
  await Promise.all(Array.from({ length: limit }, async () => {
    while (next < tasks.length) await tasks[next++]();
  }));
}

const credits = [];
// Poly Haven's CDN handles parallel downloads; SMK's server throttles them, so its files go one at a time.
const tasks = [];
const smkTasks = [];

for (const id of POLY_HAVEN) {
  const [info, files] = await Promise.all([json(`https://api.polyhaven.com/info/${id}`), json(`https://api.polyhaven.com/files/${id}`)]);
  const gltf = files.gltf['2k'].gltf;
  const dir = join(out, 'polyhaven', id);
  tasks.push(() => download(gltf.url, join(dir, `${id}.gltf`), gltf.size));
  for (const [path, file] of Object.entries(gltf.include)) tasks.push(() => download(file.url, join(dir, path), file.size));
  credits.push({
    id,
    name: info.name,
    file: `polyhaven/${id}/${id}.gltf`,
    triangles: info.polycount,
    source: `https://polyhaven.com/a/${id}`,
    authors: Object.keys(info.authors ?? {}),
    license: 'CC0 1.0',
  });
}

for (const number of SMK) {
  const result = await json(`https://api.smk.dk/api/v1/art/?object_number=${encodeURIComponent(number)}&lang=en`);
  const item = result.items?.[0];
  if (!item?.public_domain || !item.files_3D?.length) throw new Error(`SMK ${number}: no public-domain 3D file`);
  const file3d = [...item.files_3D].sort((a, b) => b.file_size - a.file_size)[0];
  const name = `${number.replace(/\W+/g, '_')}.stl`;
  // SMK's server sometimes can't serve parts of large files; a statue that fails is skipped (the scene uses the rest).
  smkTasks.push(() =>
    download(file3d.url, join(out, 'smk', name), file3d.file_size).catch((error) => {
      console.warn(`  skipping ${item.titles?.[0]?.title ?? number}: ${error.message} (run again later to retry)`);
    })
  );
  credits.push({
    id: number,
    name: item.titles?.[0]?.title ?? number,
    file: `smk/${name}`,
    bytes: file3d.file_size,
    triangles: Math.round((file3d.file_size - 84) / 50), // binary STL: 50 bytes per triangle
    source: `https://open.smk.dk/artwork/image/${encodeURIComponent(number)}`,
    authors: ['SMK, National Gallery of Denmark'],
    license: 'Public domain (CC0)',
  });
}

mkdirSync(out, { recursive: true });
writeFileSync(join(out, 'credits.json'), JSON.stringify(credits, null, 2) + '\n');

const count = tasks.length + smkTasks.length;
console.log(`Downloading ${count} files into ${out} (skipping complete ones) ...`);
let done = 0;
const counted = (task) => async () => {
  await task();
  if (++done % 10 === 0 || done === count) console.log(`  ${done}/${count}`);
};
await Promise.all([pool(tasks.map(counted), 4), pool(smkTasks.map(counted), 1)]);
const total = credits.reduce((a, c) => a + c.triangles, 0);
console.log(`Downloaded: ${credits.length} scans, ${(total / 1e6).toFixed(1)} million triangles. Credits in scans/credits.json`);

// Bake the statue scans (untextured) to .vgeo once, so the demo loads them in seconds instead of building
// 4-million-triangle hierarchies in the browser. The textured rock scans load as glTF through vg.add() (cached).
for (const credit of credits.filter((c) => c.file.endsWith('.stl') && existsSync(join(out, c.file)) && statSync(join(out, c.file)).size === c.bytes)) {
  const dir = join(out, 'baked', credit.id.replace(/\W+/g, '_'));
  if (existsSync(join(dir, 'manifest.json'))) continue;
  console.log(`Baking ${credit.name} (${(credit.triangles / 1e6).toFixed(1)}M triangles) ...`);
  const result = spawnSync(process.execPath, [join(root, 'scripts', 'bake.mjs'), join(out, credit.file), '-o', dir], {
    stdio: 'inherit',
    env: { ...process.env, NODE_OPTIONS: '--max-old-space-size=12288' },
  });
  if (result.status !== 0) throw new Error(`baking ${credit.file} failed`);
}

if (process.argv.includes('--dev')) {
  console.log('Starting the dev server: http://localhost:8090/?scene=scans');
  spawn('npx', ['vite', '--open', '/?scene=scans'], { cwd: root, stdio: 'inherit', shell: process.platform === 'win32' });
} else if (!process.argv.includes('--no-open')) {
  // The production build, served statically: what users get, and noticeably faster than the dev server.
  console.log('Building the demo ...');
  const built = spawnSync('npx', ['vite', 'build', '--logLevel', 'warn'], { cwd: root, stdio: 'inherit', shell: process.platform === 'win32' });
  if (built.status !== 0) throw new Error('building the demo failed');
  console.log('Serving it: http://localhost:8092/?scene=scans&tour');
  spawn('npx', ['vite', 'preview', '--open', '/?scene=scans&tour'], { cwd: root, stdio: 'inherit', shell: process.platform === 'win32' });
}
