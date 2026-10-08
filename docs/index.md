---
layout: home

hero:
  name: three-virtual-geometry
  text: Billions of triangles in three.js
  tagline: Virtual geometry for three.js WebGPU. Load your models, and every frame the GPU draws just the detail the screen can show. One line to set up.
  image:
    src: /screenshots/ruins-closeup.jpg
    alt: A 2-million-triangle bronze sculpture in front of a ruined temple, rendered live in the browser
  actions:
    - theme: brand
      text: Try the live demo
      link: /live
    - theme: alt
      text: Get started
      link: /guide/getting-started
    - theme: alt
      text: GitHub
      link: https://github.com/elad12390/three-virtual-geometry

features:
  - title: One line
    details: "<code>await vg.add(gltf.scene)</code> converts every static mesh of a glTF scene: materials, textures, shadows and instancing included. Then render as usual."
  - title: Automatic level of detail
    details: Each cluster of 128 triangles picks its detail on the GPU so the error stays under a pixel. No LOD meshes to author, no popping distances to tune.
  - title: Millions of instances
    details: Frustum, small-object, distance and occlusion culling run on the GPU over cells of instances. Millions of trees and rocks cost a handful of dispatches.
  - title: Works with three.js
    details: Any NodeMaterial, any light, shadows, fog, tone mapping. A VirtualMesh is a regular three.js Mesh you add to a scene.
---

<div class="paper">

## Abstract

Rendering highly detailed scenes in real time usually means hand-made levels of detail, aggressive culling and a
tight polygon budget. **Virtual geometry** removes the trade-off: every mesh is preprocessed into a hierarchy of small
triangle clusters, and every frame a compute pass selects, per cluster, the coarsest version whose error is below a
pixel on screen. Detail is then proportional to screen pixels, not to scene size.

**three-virtual-geometry** brings this technique to [three.js](https://threejs.org) on WebGPU. It builds the cluster
hierarchy (with meshoptimizer), selects and culls clusters entirely on the GPU, and draws each mesh with a single
indirect draw that works with three.js materials, lights and shadows. Foliage, which plain triangle simplification
handles poorly, gets voxel-based coarse levels automatically. A whole scene goes through shared buffers and a fixed
set of 12 compute passes, so a level with hundreds of different meshes costs the same per frame as one with a single
mesh.

<figure>
  <img src="/screenshots/ruins-wide.jpg" alt="An ancient city of 5.7 billion triangles: temples, streets, walls, hills and trees">
  <figcaption>The showcase scene: 5.7 billion full-detail triangles in 113,000 instances, where every paving stone,
  column flute and carving is real geometry. About 5 million triangles are drawn per frame.
  <a href="#live-demos">Run it below</a>.</figcaption>
</figure>

## Results

| Scene | Full-detail triangles | Instances | Drawn per frame |
| --- | --- | --- | --- |
| Ruins (showcase) | 5.7 billion | 113k | about 5M |
| World (6 km landscape) | 37.5 billion | 3 million | about 5M |
| Stress test | up to 158 billion | up to 4 million | 2 to 6M |
| glTF import (Kenney cars) | 5.1 million | 12,366 meshes → 58 | about 0.7M |

All scenes run at the display refresh rate (60 to 120 FPS) on an Apple-silicon MacBook in Chrome. Drawn triangles
stay roughly constant as scenes grow, because detail follows screen pixels.

<div class="gallery">
  <figure><img src="/screenshots/world-meshlets.jpg" alt="Debug view: every cluster in its own color"><figcaption>Clusters (debug view). Near trees use small clusters of full detail, distant hills a few large ones.</figcaption></figure>
  <figure><img src="/screenshots/world.jpg" alt="A 6 km landscape with forests, a lake and mountains"><figcaption>The same view, shaded: 3 million instances, 37.5 billion triangles.</figcaption></figure>
  <figure><img src="/screenshots/ruins-columns.jpg" alt="Fluted columns and paving stones up close"><figcaption>Up close: every flute, block and paving stone is geometry.</figcaption></figure>
  <figure><img src="/screenshots/stress.jpg" alt="A forest of a million instances"><figcaption>A million instances, culled in cells of 128 on the GPU.</figcaption></figure>
  <figure><img src="/screenshots/import.jpg" alt="A parking lot of textured glTF cars"><figcaption>glTF import: 12,366 meshes converted with one call, textures included.</figcaption></figure>
  <figure><img src="/screenshots/map.jpg" alt="A valley with conifers, a lake and mountains"><figcaption>A 2 km valley with shadows, generated at load time.</figcaption></figure>
</div>

## Live demos

Every scene runs right here, in your browser, with WebGPU (Chrome, Edge or Safari 26+). Pick one and press run. Only
the selected scene runs, and it pauses when you scroll away. The first load builds the scene's cluster hierarchies;
later loads come from the browser cache. The left panel shows what the GPU is doing, and the settings panel on the
right holds the levers.

<DemoPlayer />

## How it works

1. **Build (once per mesh, cached).** The mesh is split into clusters of up to 128 triangles. Neighbouring clusters
   are grouped, merged, simplified to half the triangles with their shared borders locked, and split again,
   level after level, into a hierarchy (a DAG). Every cluster stores its error and its parent's error, and errors
   only grow toward the root. When simplification stalls on aggregate geometry like tree crowns, a voxel proxy of the
   original takes over the coarse levels.
2. **Select (every frame, on the GPU).** A cluster is drawn when its own projected error is at most the threshold
   (about one pixel) and its parent's is above it. Because errors grow monotonically, exactly one version of every
   part of the surface passes, so there are no holes and no overlaps. Cells of instances, single instances and
   clusters are culled against the view frustum, by size, by distance and (when it pays off) by occlusion.
3. **Draw.** The selected clusters are expanded into one index buffer per pool, and each mesh draws its region
   with one indirect draw. The vertex shader pulls vertices from storage buffers, so the material is a normal
   three.js NodeMaterial.

More detail: [How it works](/guide/how-it-works).

## Use it

```ts
import { VirtualGeometry } from 'three-virtual-geometry';

const vg = new VirtualGeometry();
await vg.add(gltf.scene);   // every static mesh becomes virtual geometry
scene.add(gltf.scene);
renderer.setAnimationLoop(() => renderer.render(scene, camera));
```

[Getting started](/guide/getting-started) · [Let your AI agent set it up](/guide/ai-agents)

## Acknowledgements

This is an independent, open-source recreation of the virtual geometry technique popularized by Nanite, written
from public descriptions of the technique. It contains no code, shaders, binaries or assets from Epic Games, and
is not affiliated with or endorsed by Epic Games. "Nanite" is a trademark of Epic Games, Inc. Parts of the
preprocessing and GPU pipeline started from the MIT-licensed
[nanite-webgpu](https://github.com/Scthe/nanite-webgpu) by Marcin Matuszczyk. Mesh simplification and clustering use
[meshoptimizer](https://github.com/zeux/meshoptimizer) by Arseny Kapoulkine. Cars in the import demo are from the
[Kenney Car Kit](https://kenney.nl) (CC0).

## Citation

```bibtex
@software{benhaim2026threevirtualgeometry,
  author = {Ben-Haim, Elad},
  title  = {three-virtual-geometry: Virtual Geometry for three.js WebGPU},
  year   = {2026},
  url    = {https://github.com/elad12390/three-virtual-geometry},
  license = {MIT}
}
```

</div>

<style>
.paper { max-width: 960px; margin: 48px auto 0; padding: 0 24px; }
.paper h2 { border-top: none; margin-top: 48px; }
.paper figure { margin: 24px 0; }
.paper figure img { width: 100%; border-radius: 8px; display: block; }
.paper figcaption { font-size: 14px; color: var(--vp-c-text-2); margin-top: 8px; line-height: 1.5; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }
.gallery figure { margin: 0; }
</style>
