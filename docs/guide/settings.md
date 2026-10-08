# Settings and levers

Everything works out of the box. These are the levers when you want to change how it renders, never what.

## Quality

```ts
vg.errorThreshold.value = 1; // pixels
```

The **error threshold** is the most a cluster may deviate from the full-detail surface, in screen pixels, for it to
be drawn. Lower is sharper and costs more triangles; higher is faster.

- `1` (default): changes of detail are hard to see.
- `0.5`: for close inspection or high-end hardware.
- `2` to `3`: for weak GPUs. Detail changes start to show while moving.

It is measured in CSS pixels (`vg.errorInCssPixels = true`), so it means the same on a Retina screen and on a 1x
screen.

## Performance budget (optional)

```ts
vg.triangleBudget = 8_000_000; // 0 (default): off
vg.thresholdRange = [0.5, 3];  // how far the threshold may move
```

With a budget, the threshold adapts slowly to keep the drawn triangles near it. Off by default: a fixed threshold
gives the steadiest image, because detail then only changes with distance.

## Draw capacity

Each pool of meshes has fixed-size draw buffers. Set them when you create the `VirtualGeometry`:

```ts
const vg = new VirtualGeometry({
  maxTriangles: 10_000_000,     // camera, per pool (default 10M, 120 MB)
  maxShadowTriangles: 4_000_000, // shadow maps, per pool (default 4M)
  maxMeshlets: 1 << 20,          // drawn clusters per pool (default about 1M)
});
```

You never see an overflow: before a buffer fills up, the threshold is raised temporarily, then given back.

## Culling

| Setting | Default | What it does |
| --- | --- | --- |
| `vg.minPixelRadius.value` | `0.7` | Instances whose bounding sphere is smaller than this (pixels, radius) are not drawn. Nothing wider than about 1.4 px on screen is ever dropped. |
| `vg.frustumCulling.value` | `1` | Frustum culling of cells, instances and clusters. |
| `vg.occlusion.enabled` | `true` | Occlusion culling against solid meshes. |
| `vg.occlusion.auto` | `true` | Occlusion has a fixed cost, so it is decided by measurement: frame times with it off and on are compared every few seconds and the faster one is kept. Set `false` to keep it on. |

## Shadows

Shadows need nothing special: set `castShadow` on lights and meshes as in three.js.

| Setting | Default | What it does |
| --- | --- | --- |
| `vg.shadowErrorScale.value` | `3` | Shadow maps use the threshold times this. `1` uses camera detail. |
| `vg.autoShadowSweep` | `true` | Casters just outside the view still cast shadows into it. The light direction is read from the shadow camera. |

## Updating

| Setting | Default | What it does |
| --- | --- | --- |
| `vg.autoUpdate` | `true` | Selection and culling run before every render of a scene with virtual meshes, for the camera you render with. Set `false` and call `vg.update(renderer, camera)` yourself. |
| `vg.asyncCompile` | `true` | Compile GPU pipelines in the background (meshes appear a moment later) instead of freezing on first use. |
| `vg.freeze` | `false` | Keep the current selection, to fly around and inspect it. |

## Debug views

```ts
import { VG_DEBUG_MODES } from 'three-virtual-geometry';
vg.debugMode.value = VG_DEBUG_MODES.meshlets; // shaded, meshlets, lodLevel, instances
```

## Stats

```ts
const stats = await vg.readStats(renderer);
// { drawnTriangles, drawnMeshlets, fullDetailTriangles, instances, visibleInstances,
//   occludedInstances, shadowTriangles, capacityUse, overflow, ... }
```

`vg.lastStats` holds the latest readback (refreshed every `vg.statsInterval` frames) without awaiting.
