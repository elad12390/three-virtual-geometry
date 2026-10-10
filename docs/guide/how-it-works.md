# How it works

## 1. Preprocessing: a hierarchy of clusters

`buildVirtualMesh` turns a triangle mesh into a hierarchy (a DAG) of **meshlets**, clusters of up to 128 triangles:

1. The mesh is split into meshlets (meshoptimizer `buildMeshlets`).
2. Meshlets that share boundary edges are grouped, about 12 at a time, by a greedy region-growing partition.
3. Each group is merged and simplified to half the triangles with its outer border locked
   (meshoptimizer `simplify`), then split into meshlets again. Locking the borders means neighbouring groups always
   meet exactly, at every level.
4. Repeat until a few meshlets remain. Below that, the whole remaining mesh keeps being halved down to a few
   triangles.

Each meshlet stores its **error** (how far it may deviate from the full-detail surface) and its **parent error**
(the error of the coarser version that replaces its group). Errors only grow toward the root.

**Voxel levels for foliage.** Simplification can't reduce a tree crown made of hundreds of separate leaf clumps
without deleting clumps, so its error jumps. For coarse levels the builder also voxelizes the full-detail mesh,
closes gaps between parts, extracts a smooth surface and simplifies that, then keeps whichever candidate has the
smaller measured error. Solid meshes keep their triangles; foliage switches automatically.

**Leaf cards.** A group made of separate cards (palm leaflets, grass blades) shares no vertices between its pieces,
so edge collapses stall on it. Such a group is reduced piece by piece instead: each card is simplified with its open
edges free, then neighbouring cards are thinned in pairs, the kept card widened to cover both, so the foliage keeps
its density (stochastic simplification of aggregate detail). The error is how far the coverage moved. The builder
uses this only where it is more accurate than a voxel stand-in with as few triangles.

## 2. Selecting the cut, every frame

A meshlet is drawn when, seen from the camera, its own error is at most the threshold (about one pixel) and its
parent error is above it:

```
draw(meshlet) = projectedError(meshlet) <= threshold < projectedError(parent)
```

Because errors grow monotonically up the hierarchy, exactly one version of every part of the surface passes, so
the selected set never has holes or overlaps.

**Blending levels.** With that test alone, a meshlet switches to its parent at one exact distance, all of its
pixels at once. At about a pixel of error that is invisible, and levels switch directly. Above that (the effective
threshold rises to 2.5 px), every pixel gets its own threshold inside a band `[t, t * lodBlend]`, from a per-pixel
pattern, and shows the cut for that threshold. The selection keeps every meshlet that is in the cut for some
threshold of the band and passes the fragment shader the range of the pattern it owns, from its own error to its
parent's on a log scale. Each pixel therefore sees one complete cut, and as the camera moves detail passes from one
level to the next a few pixels at a time. With temporal anti-aliasing the pattern changes every frame and averages
into a crossfade.

The selection runs in compute shaders (three.js TSL), in a fixed set of passes per **pool** of meshes:

1. **Cells.** Instances are sorted along a Morton curve and grouped in cells of 128. One thread per cell rejects
   cells that are outside the view frustum, too small on screen, or beyond the mesh's draw distance.
2. **Instances.** One workgroup per visible cell tests each instance and computes the range of detail levels it
   can use from its distance. Instances whose few meshlets are all certain to pass emit them right away (fast path).
3. **Meshlets.** The remaining instances test each meshlet of their range: frustum, occlusion and the LOD test.
4. **Prefix.** Selected triangles are counted per mesh; one pass lays each mesh's region out in the shared index
   buffer and writes its indirect draw arguments.
5. **Expand.** One workgroup per drawn meshlet writes its triangles' indices. An index encodes
   `drawSlot * 128 + localVertex`, so vertices shared inside a meshlet hit the vertex cache.

Shadow maps get a second, coarser selection from the same passes. Casters just outside the view are kept by
sweeping their bounds along the light direction.

## 3. Drawing with three.js

Each VirtualMesh is a three.js `Mesh` with an indexed indirect draw over its region of the pool's index buffer. Its
material's `positionNode` decodes the index and pulls position, normal, UV and color from storage buffers, so
lighting, shadows, fog and tone mapping come from three.js unchanged.

## Pools: why many unique meshes stay cheap

All meshes' vertices, meshlets, level tables and instances are appended to shared storage buffers, and one set of
12 compute passes selects the cut for all of them. CPU cost and shader compilation do not grow with the number of
unique meshes. A pool is limited by WebGPU's 128 MB storage-buffer binding size; when one is full, another is
opened.

## Occlusion culling

Last frame's selection of solid meshes is drawn depth-only from the current camera into a small depth texture,
which is reduced into a hierarchical depth buffer. Instances and meshlets entirely behind it are skipped. Using last
frame's visible set as occluders is conservative: it can miss occluders, never invent them. The tested spheres are
grown by the error threshold, because occluders are simplified too. Since the extra pass has a fixed cost,
`occlusion.auto` measures frame times with it on and off and keeps the faster setting.

## Keeping the image steady

- With a fixed threshold, detail changes only when an object's distance changes, and LOD blending spreads each
  change over a band of distances.
- Draw buffers have fixed sizes. An overflow would drop meshlets in a different order every frame (holes). A
  capacity controller on the GPU checks each cut after selection: if it does not fit, it is selected again at a
  coarser threshold in the same frame (up to two retries), so no frame ever draws a partial cut. Between frames the
  factor glides by a few percent, without any CPU readback in the loop.
- The optional triangle budget moves the threshold by a few percent per frame at most.
- Meshlets inside a blend band are drawn by a separate draw with the blend mask; the rest keep a shader without
  discard, so early depth testing is not lost.
