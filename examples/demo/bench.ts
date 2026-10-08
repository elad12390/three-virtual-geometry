/**
 * Deterministic benchmark: `?bench` (optionally `&threshold=1`).
 * Visits fixed camera poses with a fixed error threshold and records GPU compute/render time
 * (timestamp queries), CPU frame time and drawn geometry. Results land in `window.__bench`.
 */
import type { DemoApp } from './app';

interface Pose {
  name: string;
  position: [number, number, number];
  target: [number, number, number];
}

const POSES: Pose[] = [
  { name: 'overview', position: [-100, 300, 300], target: [-100, 0, 500] },
  { name: 'tree-level', position: [-58, 22, 389], target: [-65, 24, 380] },
  { name: 'meadow', position: [-20, 40, 420], target: [60, 20, 200] },
  { name: 'lake-shore', position: [-120, 46, 520], target: [40, 45, 100] },
];

const withTimeout = <T>(p: Promise<T>, ms: number) =>
  Promise.race([p, new Promise<undefined>((resolve) => setTimeout(() => resolve(undefined), ms))]);

export async function runBenchmark(app: DemoApp) {
  const params = new URLSearchParams(location.search);
  const threshold = Number(params.get('threshold') ?? 1);
  // mode: full | computeOnly (skip rendering) | renderOnly (freeze LOD selection) | noShadow
  //       | vertexOnly (freeze the full-res cut, then render at 64x36 so fragments are ~free)
  const mode = params.get('mode') ?? 'full';
  if (mode === 'noShadow') {
    app.renderer.shadowMap.enabled = false;
    app.scene.traverse((o) => ((o as { castShadow?: boolean }).castShadow = false));
  }
  if (mode === 'computeOnly') (app.renderer as unknown as { render: () => void }).render = () => {};
  app.settings.autoLod = false;
  app.vg.triangleBudget = 0;
  app.vg.errorThreshold.value = threshold;
  app.controls.enableDamping = false;
  app.renderer.setAnimationLoop(null); // drive frames manually

  const results: Record<string, unknown>[] = [];

  for (const pose of POSES) {
    app.camera.position.set(...pose.position);
    app.controls.target.set(...pose.target);
    app.controls.update();
    console.log('BENCH pose', pose.name);
    const device = (app.renderer.backend as unknown as { device: GPUDevice }).device;
    const run = async (n: number) => {
      for (let i = 0; i < n; i++) {
        app.renderFrame();
        if (i % 8 === 7) await device.queue.onSubmittedWorkDone(); // keep the queue shallow
      }
      await device.queue.onSubmittedWorkDone();
    };
    app.settings.freezeLOD = false;
    await run(16); // warm up (pipelines, LOD settles)
    if (mode === 'renderOnly' || mode === 'vertexOnly') app.settings.freezeLOD = true;
    if (mode === 'vertexOnly') app.renderer.setSize(64, 36, false);
    const frames = 96;
    const t0 = performance.now();
    await run(frames);
    const t1 = performance.now();
    if (mode === 'vertexOnly') app.renderer.setSize(window.innerWidth, window.innerHeight, false);
    const stats = await app.vg.readStats(app.renderer);
    const perMesh: Record<string, number> = {};
    for (const m of app.vg.meshes) perMesh[m.name] = Math.round((await m.readStats(app.renderer)).drawnTriangles / 1000);
    results.push({
      pose: pose.name,
      mode,
      threshold,
      frameMs: +((t1 - t0) / frames).toFixed(2),
      drawnTriangles: stats.drawnTriangles,
      drawnMeshlets: stats.drawnMeshlets,
      kTrisPerMesh: JSON.stringify(perMesh),
    });
  }
  (window as unknown as { __bench: unknown }).__bench = results;
  console.log('BENCH', JSON.stringify(results));
  return results;
}
