# Meshes and instances

A `VirtualMesh` is one preprocessed geometry drawn with any number of instances. It is a regular three.js `Mesh`:
add it to a scene, set `castShadow`, `receiveShadow`, `visible`, `name` and `userData` as usual.

## Creating one

```ts
import { buildVirtualMeshCached, fromBufferGeometry } from 'three-virtual-geometry';

const data = await buildVirtualMeshCached(fromBufferGeometry(geometry));
const mesh = vg.createMesh(data, material, {
  matrices,          // Float32Array, 16 floats (column-major Matrix4) per instance
  colors,            // optional Float32Array, rgba per instance, multiplied with the material color
}, {
  maxDrawDistance: 500, // optional
});
scene.add(mesh);
```

Instance matrices are in **world space**. Keep the VirtualMesh object itself (and its parents) at the origin, with no
rotation or scale, and place instances only through their matrices: culling and level of detail use the matrices, so
a transform on the mesh object would make them cull the wrong places.

`fromBufferGeometry` welds vertices (keeping UV seams), computes smooth normals and keeps UVs and vertex colors. If
you already have arrays, pass a `VirtualMeshSource` (`positions`, `normals`, `indices`, optional `uvs` and
`colors`) straight to `buildVirtualMeshCached`.

## Moving instances

```ts
const m = new THREE.Matrix4();
for (const i of movedInstances) {
  m.compose(position[i], rotation[i], scale[i]);
  mesh.setMatrixAt(i, m);
}
mesh.commitInstances(); // once per batch: uploads the changes
```

Instances can move every frame. Only the changed groups of 128 instances get new culling bounds.

## Per-mesh levers

| Property | Default | What it does |
| --- | --- | --- |
| `mesh.maxDrawDistance` | `Infinity` | Instances farther than this are not drawn. They shrink to nothing over the last 15% of the distance, so they do not pop. Great for grass and small props. |
| `mesh.minPixelRadius` | `0` | Hide instances smaller than this many pixels (radius) on screen. The larger of this and the global value applies. |
| `mesh.occluder` | solid materials | Draw this mesh into the occlusion pass, so it can hide what is behind it. |
| `mesh.occlusionCulling` | `true` | Let occlusion culling hide this mesh's instances. |
| `mesh.castShadow` | `false` | As in three.js. Shadows use their own, coarser selection automatically. |

## Removing

```ts
mesh.dispose(); // stops rendering it and frees its slot
```

## Vertex colors and tints

Per-vertex colors come from the source's `colors` (or a `color` attribute in glTF). Per-instance colors come from
the `colors` array. Both multiply the material color, so a white material shows them as they are.
