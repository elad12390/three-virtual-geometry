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

## Real scans

The other demo scenes are generated in code, so they load in seconds from a web page. Virtual geometry exists for a
different kind of data: real-world scans, captured by 3D scanners or photogrammetry at millions of triangles per
object, used as they are, without anyone making simplified versions by hand. The **real scans** scene shows them as
exhibits in a museum:

| Exhibits | Data | Source and license |
| --- | --- | --- |
| 723 statues in the galleries and the rotunda, 5 of them monumental on the hero bases | 14 scans of plaster casts of classical sculpture (Herakles, Apollo Lykeios, the Dying Gladiator, the Wrestlers, Minerva, Thalia, Sophocles, ...), 1 to 4 million triangles each: 2.1 billion triangles in all | [SMK, National Gallery of Denmark](https://open.smk.dk), public domain |
| 40 specimens in the glazed Great Court | 13 photoscanned coastal rocks and cliffs, 1 to 2.9 million triangles each, with 2K textures | [Poly Haven](https://polyhaven.com/models/rocks), CC0 |

The building itself (rotunda and coffered dome, colonnaded galleries, glazed court) is generated in code, like the
other demo scenes. Each frame draws about 0.4 to 1.5 million of those 2.1 billion triangles, at one pixel of error.

### Why it's a video on this site

Together the scans are about **2 GB**. Downloading that just by opening a web page would be unreasonable, so the
[live demo page](/live#scans) shows a recording instead. The recording is the real thing: the same scene rendered
live on a MacBook Pro (M4 Pro) in Chrome, recorded in real time at 1920×1080 with the stats panel showing. The frame rate
is uncapped: each frame starts as soon as the GPU has finished the last one, and the panel shows that rate (a little
lower than without recording, since the same machine is also encoding the video). The video itself is 60 frames per second.

### Run it yourself

```bash
git clone https://github.com/elad12390/three-virtual-geometry
cd three-virtual-geometry
npm install
npm run demo:scans                # downloads the scans once (about 2 GB), bakes the statues, opens the scene
npm run demo:scans -- --small     # a quicker subset: 3 rock scans and 2 statues
npm run demo:scans -- --dev       # open it on the dev server (live reload) instead of the production build
```

`npm run demo:scans` downloads into `examples/scans/` (not part of the repository), resumes interrupted downloads,
and skips everything already there, so running it again only opens the scene. It then builds the demo and serves the
production build at `http://localhost:8092`, as on this site; the dev server is noticeably slower. If a statue
fails to download (the museum's server sometimes drops large transfers), the scene uses the others, and running the
command again picks up where it stopped. The statues are baked to
`.vgeo` files with `vg-bake` once. The textured rock scans load as glTF through `vg.add()`, so the first visit builds
them in the browser (a few minutes) and later visits load them from the cache. In the scene, `&tour` (or the
**fly-through** button) plays the fly-through, a 214-second loop that circles the statues at eye level, spirals up around the
rotunda's giant and crosses the court before returning to where it started.
