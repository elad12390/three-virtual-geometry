# Performance

## What costs what

| Cost | Grows with | Lever |
| --- | --- | --- |
| Drawing | screen pixels and the threshold, not scene size | `errorThreshold`, `triangleBudget` |
| Selection (compute) | visible instances | `maxDrawDistance`, `minPixelRadius`, occlusion |
| CPU | number of unique meshes (draw calls only) | nothing to do |
| Load time | source triangles | the build cache, or baked `.vgeo` files |
| GPU memory | source triangles and instances | `maxTriangles` and the other capacities |

## Tips

- **Give dense meshes.** Virtual geometry shines with detailed sources. A 100k-triangle rock costs about the same
  as a 1k one once it is a few dozen pixels tall.
- **Instance everything you can.** `vg.add` merges repeated geometry automatically. With `createMesh`, put all
  copies of an asset in one mesh.
- **Set a draw distance on small clutter.** Grass, pebbles and flowers at 300 to 500 m cost almost nothing beyond
  that: whole cells of 128 instances are rejected at once, and instances shrink away instead of popping.
- **Use `prune: true` for thin geometry** (grass, leaves) when building.
- **Shadows** already use a coarser selection (`shadowErrorScale`). Turn `castShadow` off for small clutter.
- **On weak GPUs,** raise `errorThreshold` to 1.5 to 2, or set a `triangleBudget`.

## Measured

On an Apple-silicon MacBook in Chrome, at the display refresh rate (60 to 120 FPS):

| Scene | Full-detail triangles | Instances | Drawn |
| --- | --- | --- | --- |
| Ruins | 5.7 billion | 113k | about 5M |
| World | 37.5 billion | 3 million | about 5M |
| Stress, 1 million instances | 39 billion | 1 million | about 3M |

Run the stress test yourself with `?scene=stress&count=N` in the demo (up to 4 million).
