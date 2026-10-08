// Uncapped FPS benchmark of the demo scenes in Chrome, written to docs/benchmarks.json (shown on the Performance page).
//
//   npm run bench                               # build the demo, run every scene at 1080p and 4K
//   npm run bench -- --scenes ruins,map         # only some scenes
//   npm run bench -- --url https://elad12390.github.io/three-virtual-geometry/demo/   # bench a deployed demo
//   npm run bench -- --headed                   # show the browser window
//
// Each scene runs with `?bench`: fixed camera views, fixed 1 px error threshold, frames submitted back to back with
// no vsync (see examples/demo/bench.ts). Needs Google Chrome installed (Playwright drives it, no extra download).
import { chromium } from 'playwright';
import { build, preview } from 'vite';
import { writeFileSync } from 'node:fs';
import { cpus, totalmem, platform, release } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const arg = (name) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : undefined;
};

const SCENES = {
  ruins: 'scene=ruins',
  world: 'scene=world',
  map: 'scene=map',
  import: 'scene=import',
  stress: 'scene=stress&count=1000000',
};
const RESOLUTIONS = [
  { name: '1080p', width: 1920, height: 1080, scale: 1 },
  { name: '4K', width: 1920, height: 1080, scale: 2 },
];
const scenes = (arg('scenes') ?? Object.keys(SCENES).join(',')).split(',');

let base = arg('url');
let server;
if (!base) {
  await build({ configFile: join(root, 'vite.config.ts'), logLevel: 'warn' });
  server = await preview({ configFile: join(root, 'vite.config.ts'), preview: { port: 4319, strictPort: true } }); // fixed port: the build cache is per origin
  base = server.resolvedUrls.local[0];
}

// A persistent profile keeps the IndexedDB build cache between runs, so only the first run builds the scenes.
const profile = join(root, 'node_modules', '.cache', 'vg-bench-profile');
const runs = [];
let browserVersion = '';
let gpu = '';

for (const res of RESOLUTIONS) {
  const context = await chromium.launchPersistentContext(profile, {
    channel: 'chrome',
    headless: !args.includes('--headed'),
    viewport: { width: res.width, height: res.height },
    deviceScaleFactor: res.scale,
    args: ['--enable-unsafe-webgpu', '--disable-gpu-vsync', '--disable-frame-rate-limit'],
  });
  browserVersion = context.browser()?.version() ?? browserVersion;
  for (const scene of scenes) {
    const page = await context.newPage();
    page.on('pageerror', (e) => console.error(`  [${scene}] ${e.message}`));
    const t0 = Date.now();
    await page.goto(`${base}?${SCENES[scene]}&bench`);
    await page.waitForFunction(() => window.__bench, null, { timeout: 20 * 60_000, polling: 1000 });
    const report = await page.evaluate(() => window.__bench);
    gpu = report.gpu;
    runs.push({ scene, resolution: res.name, width: report.width, height: report.height, results: report.results });
    console.log(`${scene} @ ${res.name} (${report.width}x${report.height}, ${((Date.now() - t0) / 1000).toFixed(0)} s)`);
    for (const r of report.results) {
      console.log(
        `  ${r.pose.padEnd(16)} ${String(r.fps).padStart(5)} FPS  ${r.frameMs.toFixed(2).padStart(6)} ms  ` +
          `cpu ${r.cpuMs.toFixed(2)} ms  drawn ${(r.drawnTriangles / 1e6).toFixed(2)}M  shadow ${(r.shadowTriangles / 1e6).toFixed(2)}M  ` +
          `visible ${r.visibleInstances}  rounds ${r.roundsMs.join('/')}  occl ${r.occlusionShare}`
      );
    }
    await page.close();
  }
  await context.close();
}
await server?.close();

const out = {
  date: new Date().toISOString().slice(0, 10),
  machine: `${cpus()[0].model}, ${Math.round(totalmem() / 2 ** 30)} GB, ${platform()} ${release()}`,
  browser: `Chrome ${browserVersion}`,
  gpu,
  source: arg('url') ?? 'local production build',
  runs,
};
if (args.includes('--no-write')) console.log(JSON.stringify(out, null, 2));
else {
  writeFileSync(join(root, 'docs', 'benchmarks.json'), JSON.stringify(out, null, 2) + '\n');
  console.log('wrote docs/benchmarks.json');
}
