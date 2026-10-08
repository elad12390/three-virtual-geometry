// Builds the two single-file CDN bundles (served by jsDelivr and unpkg straight from the npm package):
//
//   dist/three-virtual-geometry.min.js      the library with meshoptimizer built in; three.js is NOT included
//                                           (imports 'three/webgpu' and 'three/tsl': map them with an import map)
//   dist/three-virtual-geometry.all.min.js  everything in one file: the library, three.js (WebGPU build + TSL),
//                                           meshoptimizer, GLTFLoader, OrbitControls and RoomEnvironment
import * as esbuild from 'esbuild';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const { version } = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
const require = createRequire(import.meta.url);
const threeWebgpu = join(dirname(require.resolve('three')), 'three.webgpu.js');

const banner = `/**
 * three-virtual-geometry ${version} | MIT License | https://github.com/elad12390/three-virtual-geometry
 * Independent open-source implementation of virtual geometry; not affiliated with or endorsed by Epic Games.
 * Includes meshoptimizer (MIT, Arseny Kapoulkine)%THREE%. Parts derived from nanite-webgpu (MIT, Marcin Matuszczyk).
 */`;

const common = { bundle: true, format: 'esm', target: 'es2022', minify: true, sourcemap: true, sourcesContent: false, legalComments: 'none', logLevel: 'warning' };

await esbuild.build({
  ...common,
  entryPoints: [join(root, 'src/index.ts')],
  outfile: join(root, 'dist/three-virtual-geometry.min.js'),
  external: ['three', 'three/*'],
  banner: { js: banner.replace('%THREE%', '') },
});

await esbuild.build({
  ...common,
  entryPoints: [join(root, 'scripts/cdn/all.ts')],
  outfile: join(root, 'dist/three-virtual-geometry.all.min.js'),
  banner: { js: banner.replace('%THREE%', ' and three.js (MIT, three.js authors)') },
  plugins: [
    {
      // Addons import 'three'; resolve it to the WebGPU build so the bundle contains a single copy of three.js.
      name: 'three-webgpu',
      setup(build) {
        build.onResolve({ filter: /^three$/ }, () => ({ path: threeWebgpu }));
      },
    },
  ],
});
