# Set it up with your AI agent

Coding agents (Claude Code, Cursor, GitHub Copilot, Codex, Windsurf, ...) can add three-virtual-geometry to your
project for you. Paste the prompt below into your agent, in your project's folder.

## The prompt

```text
Add three-virtual-geometry (https://github.com/elad12390/three-virtual-geometry) to this project so that every
static mesh is rendered as virtual geometry (automatic level of detail and GPU culling).

Read the docs first: https://elad12390.github.io/three-virtual-geometry/llms.txt

Then:
1. Install it: npm install three-virtual-geometry (three.js r180+ is a peer dependency).
2. Make sure the renderer is WebGPU: import * as THREE from 'three/webgpu', use THREE.WebGPURenderer and
   await renderer.init(). Switch classic three.js imports to 'three/webgpu' where needed.
3. Create one VirtualGeometry for the app: const vg = new VirtualGeometry().
4. After loading each model (e.g. gltf.scene), call await vg.add(model) before adding it to the scene.
   For geometry built in code, use buildVirtualMeshCached(fromBufferGeometry(geometry)) and vg.createMesh(...).
5. Keep the render loop as it is: level of detail and culling run automatically before each render.
6. If there is a loading screen, call await vg.compileAsync(renderer, scene, camera) before hiding it.
7. Do not change what is rendered: keep the existing materials, lights and shadows.
Run the app and confirm there are no console errors.
```

## What the agent reads

- **[llms.txt](https://elad12390.github.io/three-virtual-geometry/llms.txt)**: a compact, single-page summary of the
  API and the rules above, written for language models.
- **[AGENTS.md](https://github.com/elad12390/three-virtual-geometry/blob/main/AGENTS.md)** in the repository: how
  the code is organized and how to build, test and run the demos, for agents working on the library itself.

## Checking the result

Ask your agent to log `await vg.readStats(renderer)` once after loading. `drawnTriangles` should be far below
`fullDetailTriangles` for any large scene, and `overflow` should be `false`.
