# Baking and caching

Building a mesh's cluster hierarchy takes time: about 0.3 s for a 50k-triangle tree, a few seconds for a
multi-million-triangle mesh. There are two ways to pay that only once.

## Automatic: the browser cache (default)

`vg.add` and `buildVirtualMeshCached` store every build in IndexedDB, keyed by a hash of the source data and the
build options. The next page load decodes the stored copy instead of building, typically 15 to 25 times faster.

```ts
import { buildVirtualMeshCached, clearVirtualMeshCache } from 'three-virtual-geometry';

const data = await buildVirtualMeshCached(source);                            // same as buildVirtualMesh, cached
const data2 = await buildVirtualMeshCached(source, { cache: { key: 'rock-v1' } }); // fixed key: skips hashing
await clearVirtualMeshCache();                                                 // empty it
```

- If IndexedDB is unavailable (some private modes, quota errors), it just builds. The cache never throws.
- Least recently used entries are evicted above 512 MB.
- Entries are invalidated automatically when the library's build algorithm changes.
- With a fixed `cache.key`, change the key yourself when the source or the options change.

## Offline: bake files and ship them

Bake models ahead of time with the CLI that comes with the package:

```bash
npx vg-bake model.glb -o public/baked/model   # also .gltf and .obj; flags: --prune, --no-voxel, --no-deflate
```

It writes one `.vgeo` file per unique geometry and a `manifest.json` with each file's instance transforms and
basic material values. Load them in the app:

```ts
import { loadVirtualMesh } from 'three-virtual-geometry';

const manifest = await (await fetch('baked/model/manifest.json')).json();
for (const entry of manifest.meshes) {
  const data = await loadVirtualMesh(`baked/model/${entry.file}`);
  const matrices = new Float32Array(entry.instances.flat());
  const material = new THREE.MeshStandardNodeMaterial({ color: entry.material?.color ?? 0xffffff });
  scene.add(vg.createMesh(data, material, { matrices }));
}
```

The baker reads geometry only: it skips textures and UVs, recomputes smooth normals and drops vertex colors. For
textured models, use `vg.add` with the browser cache, or bake your own sources with `encodeVirtualMesh(data)` and
load them with `loadVirtualMesh`.

## The `.vgeo` format

A small binary container: a 16-byte prefix (magic `VGEO`, format version, header size, checksum), a JSON header
describing every field, and a 4-byte-aligned payload. Compression is lossless (meshoptimizer's vertex and index
codecs, plus deflate where it helps), and files are 3 to 6 times smaller than the data in memory. Damaged files or
files from a newer format version are rejected with a `VirtualMeshFormatError`.
