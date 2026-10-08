# Importing models (glTF)

`vg.add(object)` converts every static mesh under `object`, which is usually `gltf.scene`, in place:

```ts
const gltf = await new GLTFLoader().loadAsync('level.glb');
const result = await vg.add(gltf.scene);
scene.add(gltf.scene);
```

Any `Object3D` works: a glTF scene, an FBX import, or a group you built yourself.

## Options

All options are optional:

```ts
const result = await vg.add(gltf.scene, {
  build: { prune: true },                        // build options for every geometry (see below)
  mesh: { maxDrawDistance: 400 },                // VirtualMesh options, or (group) => options
  filter: (mesh) => !mesh.name.startsWith('UI'), // return false to leave a mesh as it is
  replace: true,                                 // false: change nothing, add result.object yourself
  onProgress: (f) => (bar.style.width = `${f * 100}%`),
});
```

| Option | Default | What it does |
| --- | --- | --- |
| `build` | `{}` | Build options for every geometry. `prune: true` helps thin geometry like grass and leaves. |
| `mesh` | `{}` | Options for each created VirtualMesh, e.g. `maxDrawDistance`. A function gets the group and returns options per mesh. |
| `filter` | all meshes | Return `false` to keep a mesh as a normal three.js mesh. |
| `replace` | `true` | Hide the originals and add the converted meshes to `object`. With `false`, nothing in your scene changes. |
| `onProgress` | none | Called with 0..1 while building. Can be async, e.g. to let the page repaint. |
| `material` | automatic | Your own material conversion: `(material, group) => NodeMaterial`. |
| `builder` | cached build | The function that builds a geometry's hierarchy, e.g. `buildVirtualMesh` to skip the cache. |

## The result

```ts
result.meshes;            // the created VirtualMeshes
result.stats;             // { sourceMeshes, instances, uniqueGeometries, buildMs }
result.skipped;           // meshes left alone, and why (skinned, morph targets, unsupported material, ...)
result.syncTransforms();  // after moving or animating the source nodes
result.dispose();         // remove the VirtualMeshes and bring back the originals
```

Instances are baked in world space at import. If you move `gltf.scene` or its children afterwards, call
`result.syncTransforms()`.

## Materials and textures

Any `NodeMaterial` works. Classic materials (`MeshStandardMaterial`, `MeshPhysicalMaterial`, Phong, Lambert, Basic,
Toon, Matcap, Normal) are converted to their node equivalents by `toNodeMaterial`, which `vg.add` calls for you.

Supported: color, normal and bump maps (tangents are computed from screen-space derivatives, so no tangent
attribute is needed), roughness, metalness, emissive, AO, alpha maps, alpha test and transparency (shadows cut
holes correctly), vertex colors and instance colors.

In your own node materials, use these nodes instead of the geometry's attributes:

```ts
import { vgUv, vgWorldNormal } from 'three-virtual-geometry';
import { texture } from 'three/tsl';

material.colorNode = texture(myMap, vgUv).mul(vgWorldNormal.y.mul(0.5).add(0.5));
```

`texture(t)` without explicit UVs also samples `vgUv` automatically.

## What is not converted

- Skinned meshes, morph targets, points, lines and sprites: they keep rendering as normal three.js objects.
- Materials without a node equivalent (`ShaderMaterial`, `RawShaderMaterial`): pass a `material` option to convert
  them yourself, or they stay as normal meshes.
- Only the first UV channel is used.
