// Offline baking CLI: node scripts/bake.mjs model.glb|model.gltf|model.obj [-o outDir] [--prune] [--no-voxel] [--no-deflate]
// Bundles scripts/bake.ts (three and meshoptimizer resolved from node_modules) and runs it in Node.
import * as esbuild from 'esbuild';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dir = mkdtempSync(join(tmpdir(), 'three-virtual-geometry-bake-'));
const out = join(dir, 'bake.mjs');
await esbuild.build({ entryPoints: [join(here, 'bake.ts')], bundle: true, platform: 'node', format: 'esm', target: 'node18', outfile: out, logLevel: 'error' });
const { status } = spawnSync(process.execPath, [out, ...process.argv.slice(2)], { stdio: 'inherit' });
rmSync(dir, { recursive: true, force: true });
process.exit(status ?? 1);
