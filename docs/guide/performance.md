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

## Benchmarks

Uncapped frame rates of the demo scenes: frames are submitted back to back with no vsync, and the GPU is waited on
after every batch of 8, so these are the frame rates the hardware sustains, not the display's refresh rate. Each
scene is measured from several fixed camera views at a 1 px error threshold with shadows on, at 1920×1080 and at
3840×2160.

<BenchTable detail />

### Run them yourself

```bash
npm run bench                          # every scene, 1080p and 4K; writes docs/benchmarks.json (this table)
npm run bench -- --scenes ruins,map    # some scenes
npm run bench -- --url https://elad12390.github.io/three-virtual-geometry/demo/   # the deployed demo
```

It drives your installed Google Chrome with Playwright. In a browser you can also open any demo scene with
`?bench` appended and read `window.__bench`, or tick **uncapped fps** in its settings panel to see the uncapped frame
rate live.
