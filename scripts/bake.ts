/**
 * Offline baking, run through scripts/bake.mjs (which bundles this file for Node):
 *
 *   node scripts/bake.mjs model.glb|model.gltf|model.obj [-o outDir] [--prune] [--no-voxel] [--no-deflate]
 *
 * Writes one .vgeo file per unique geometry plus manifest.json (files, instance transforms, basic material
 * values). Geometry only: textures and UVs are skipped, vertex colors are not carried over, normals are
 * recomputed (smooth) by fromBufferGeometry. Draco-compressed glTF is not supported (meshopt-compressed is).
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname, extname, join, resolve } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { MeshoptDecoder } from 'meshoptimizer';
import { buildVirtualMesh, encodeVirtualMesh, fromBufferGeometry, VG_FORMAT_VERSION, type VirtualMeshBuildOptions } from '../src/index';

const USAGE = 'usage: npx vg-bake <model.glb|model.gltf|model.obj> [-o outDir] [--prune] [--no-voxel] [--no-deflate]';

function parseArgs(argv: string[]) {
  const args = { input: '', outDir: '', build: {} as VirtualMeshBuildOptions, deflate: true };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '-o' || a === '--out') args.outDir = argv[++i] ?? '';
    else if (a === '--prune') args.build.prune = true;
    else if (a === '--no-voxel') args.build.voxelLods = false;
    else if (a === '--no-deflate') args.deflate = false;
    else if (a === '-h' || a === '--help') {
      console.log(USAGE);
      process.exit(0);
    } else if (a.startsWith('-')) throw new Error(`unknown option ${a}\n${USAGE}`);
    else args.input = a;
  }
  if (!args.input) throw new Error(USAGE);
  args.outDir ||= `${basename(args.input, extname(args.input))}.vg`;
  return args;
}

/** Node's fetch has no file: URLs; three's FileLoader uses fetch (and ProgressEvent) for external .bin buffers. */
function serveFileUrls() {
  const g = globalThis as Record<string, unknown>;
  g.ProgressEvent ??= class extends Event {
    constructor(type: string, init: object = {}) {
      super(type);
      Object.assign(this, init);
    }
  };
  const nodeFetch = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = input instanceof Request ? input.url : String(input);
    if (!url.startsWith('file:')) return nodeFetch(input, init);
    return new Response(readFileSync(fileURLToPath(url)));
  }) as typeof fetch;
}

async function load(path: string): Promise<THREE.Object3D> {
  const ext = extname(path).toLowerCase();
  if (ext === '.obj') return new OBJLoader().parse(readFileSync(path, 'utf8'));
  if (ext !== '.glb' && ext !== '.gltf') throw new Error(`unsupported file type ${ext} (expected .glb, .gltf or .obj)`);
  serveFileUrls();
  const loader = new GLTFLoader();
  loader.setMeshoptDecoder(MeshoptDecoder);
  // Geometry only: drop every texture reference from the materials before parsing (no image decoding in Node).
  loader.register(
    (parser) =>
      ({
        name: 'three-virtual-geometry-skip-textures',
        beforeRoot: () => {
          for (const material of (parser.json.materials ?? []) as object[]) stripTextures(material);
          return null;
        },
      }) as never
  );
  const file = readFileSync(path);
  const data = ext === '.gltf' ? file.toString('utf8') : file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  const resourcePath = pathToFileURL(dirname(resolve(path))).href + '/';
  const gltf = await new Promise<{ scene: THREE.Object3D }>((ok, fail) => loader.parse(data as ArrayBuffer, resourcePath, ok, fail));
  return gltf.scene;
}

function stripTextures(node: object) {
  const record = node as Record<string, unknown>;
  for (const key of Object.keys(record)) {
    if (key.endsWith('Texture')) delete record[key];
    else if (record[key] && typeof record[key] === 'object') stripTextures(record[key] as object);
  }
}

function materialInfo(material: THREE.Material | THREE.Material[]) {
  const m = (Array.isArray(material) ? material[0] : material) as THREE.MeshStandardMaterial | undefined;
  if (!m) return null;
  return {
    name: m.name || undefined,
    color: m.color ? `#${m.color.getHexString()}` : undefined,
    roughness: m.roughness === undefined ? undefined : Number(m.roughness.toFixed(4)),
    metalness: m.metalness === undefined ? undefined : Number(m.metalness.toFixed(4)),
    doubleSided: m.side === THREE.DoubleSide || undefined,
  };
}

const slug = (s: string) => s.replace(/[^a-zA-Z0-9_-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 48) || 'mesh';

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const t0 = performance.now();
  const root = await load(args.input);
  root.updateMatrixWorld(true);

  const byGeometry = new Map<THREE.BufferGeometry, { name: string; material: THREE.Material | THREE.Material[]; instances: number[][] }>();
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry?.attributes.position) return;
    if ((mesh as unknown as THREE.Points).isPoints || (mesh as unknown as THREE.Line).isLine) return;
    let entry = byGeometry.get(mesh.geometry);
    if (!entry) byGeometry.set(mesh.geometry, (entry = { name: mesh.name || mesh.geometry.name, material: mesh.material, instances: [] }));
    const instanced = mesh as THREE.InstancedMesh;
    if (instanced.isInstancedMesh) {
      const m = new THREE.Matrix4();
      for (let i = 0; i < instanced.count; i++) {
        instanced.getMatrixAt(i, m);
        entry.instances.push(m.premultiply(mesh.matrixWorld).toArray());
      }
    } else {
      entry.instances.push(mesh.matrixWorld.toArray());
    }
  });
  if (byGeometry.size === 0) throw new Error(`no triangle meshes found in ${args.input}`);
  console.log(`${args.input}: ${byGeometry.size} unique geometries (loaded in ${(performance.now() - t0).toFixed(0)} ms)`);

  mkdirSync(args.outDir, { recursive: true });
  const meshes = [];
  let index = 0;
  for (const [geometry, entry] of byGeometry) {
    const file = `${index++}-${slug(entry.name)}.vgeo`;
    const source = fromBufferGeometry(geometry);
    if (source.indices.length < 3) {
      console.log(`  ${file}: skipped (no triangles)`);
      continue;
    }
    const data = await buildVirtualMesh(source, args.build);
    const t = performance.now();
    const bytes = await encodeVirtualMesh(data, { deflate: args.deflate });
    const encodeMs = performance.now() - t;
    writeFileSync(join(args.outDir, file), bytes);
    const s = data.stats;
    console.log(
      `  ${file}: ${s.leafTriangles} triangles, ${data.meshletCount} meshlets, ${s.lodLevels} LODs, ` +
        `built in ${s.buildMs.toFixed(0)} ms, encoded in ${encodeMs.toFixed(0)} ms, ${(bytes.length / 1024).toFixed(0)} KB, ${entry.instances.length} instance(s)`
    );
    meshes.push({
      name: entry.name,
      file,
      bytes: bytes.length,
      triangles: s.leafTriangles,
      meshlets: data.meshletCount,
      lodLevels: s.lodLevels,
      material: materialInfo(entry.material),
      instances: entry.instances,
    });
  }

  const manifest = { generator: 'three-virtual-geometry', formatVersion: VG_FORMAT_VERSION, source: basename(args.input), build: args.build, meshes };
  // One line per instance matrix.
  const json = JSON.stringify(manifest, (k, v) => (k === 'instances' ? v.map((m: number[]) => JSON.stringify(m)) : v), 2).replace(/"(\[[^"]*\])"/g, '$1');
  writeFileSync(join(args.outDir, 'manifest.json'), json);
  console.log(`wrote ${meshes.length} .vgeo files and manifest.json to ${args.outDir} in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
}

main().catch((e) => {
  console.error(e instanceof Error ? e.message : e);
  process.exit(1);
});
