/**
 * Deterministic, uncapped benchmark: `?bench` (optionally `&threshold=1`, `&mode=...`).
 * Visits fixed camera poses with a fixed error threshold and renders frames back to back (no vsync), waiting for
 * the GPU after every batch, so frame time is what the GPU and CPU can actually sustain. Results land in
 * `window.__bench` (read by scripts/bench.mjs).
 */
import * as THREE from 'three/webgpu';
import type { DemoApp } from './app';

export interface BenchPose {
  name: string;
  position: [number, number, number];
  target: [number, number, number];
}

/** Hand-placed views of the valley map. */
const MAP_POSES: BenchPose[] = [
  { name: 'overview', position: [-100, 300, 300], target: [-100, 0, 500] },
  { name: 'tree-level', position: [-58, 22, 389], target: [-65, 24, 380] },
  { name: 'meadow', position: [-20, 40, 420], target: [60, 20, 200] },
  { name: 'lake-shore', position: [-120, 46, 520], target: [40, 45, 100] },
];

const BATCH = 8;
const WARMUP_BATCHES = 4;
/** Each view is measured in ROUNDS separate rounds of BATCHES batches; the reported time is the median round. */
const ROUNDS = 3;
const BATCHES = 12;

/** The scene's starting view, and the same view pulled back and up to see much more of the scene. */
function defaultPoses(app: DemoApp): BenchPose[] {
  const position = app.camera.position.clone();
  const target = app.controls.target.clone();
  const offset = position.clone().sub(target);
  const aerial = target.clone().addScaledVector(offset, 2);
  aerial.y += offset.length();
  const v = (p: THREE.Vector3) => p.toArray() as [number, number, number];
  return [
    { name: 'start', position: v(position), target: v(target) },
    { name: 'aerial', position: v(aerial), target: v(target) },
  ];
}

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

export async function runBenchmark(app: DemoApp, sceneName: string) {
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
  app.pause(); // frames are driven manually below

  const poses = app.benchPoses ?? (sceneName === 'map' ? MAP_POSES : defaultPoses(app));
  const device = (app.renderer.backend as unknown as { device: GPUDevice }).device;
  const results: Record<string, unknown>[] = [];

  // One batch: BATCH frames submitted back to back, then wait for the GPU to finish them. Returns wall-clock ms per
  // frame and the CPU time spent per frame on the main thread.
  const batch = async () => {
    let cpu = 0;
    const t0 = performance.now();
    for (let i = 0; i < BATCH; i++) {
      const c0 = performance.now();
      app.renderFrame();
      cpu += performance.now() - c0;
    }
    await device.queue.onSubmittedWorkDone();
    return { ms: (performance.now() - t0) / BATCH, cpu: cpu / BATCH };
  };

  for (const pose of poses) {
    app.camera.position.set(...pose.position);
    app.controls.target.set(...pose.target);
    app.controls.update();
    app.settings.freezeLOD = false;
    for (let i = 0; i < WARMUP_BATCHES; i++) await batch(); // pipelines and LOD settle
    // Occlusion culling is switched on or off by measurement (occlusion.auto). Make it decide for this view now and
    // wait for the decision, so the measured frames run in steady state with the setting a user would get here.
    const tuner = app.vg.occlusion as unknown as { auto: boolean; enabled: boolean; phase: string; phaseFrames: number; autoHoldFrames: number };
    if (tuner.auto && tuner.enabled) {
      tuner.phase = 'hold';
      tuner.phaseFrames = tuner.autoHoldFrames;
      for (let i = 0; i < 2 && tuner.phase === 'hold'; i++) await batch(); // starts probing
      for (let i = 0; i < 64 && tuner.phase !== 'hold'; i++) await batch(); // probes off, then on, then holds
    }
    if (mode === 'renderOnly' || mode === 'vertexOnly') app.settings.freezeLOD = true;
    if (mode === 'vertexOnly') app.renderer.setSize(64, 36, false);
    const samples: { ms: number; cpu: number; occlusion: boolean }[] = [];
    const rounds: number[] = [];
    for (let round = 0; round < ROUNDS; round++) {
      const roundTimes: number[] = [];
      for (let i = 0; i < BATCHES; i++) {
        const s = await batch();
        samples.push({ ...s, occlusion: app.vg.occlusion.active });
        roundTimes.push(s.ms);
      }
      rounds.push(median(roundTimes));
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    if (mode === 'vertexOnly') app.renderer.setSize(window.innerWidth, window.innerHeight, false);
    const times = samples.map((s) => s.ms).sort((a, b) => a - b);
    const frameMs = median(rounds);
    const stats = await app.vg.readStats(app.renderer);
    results.push({
      pose: pose.name,
      frameMs: +frameMs.toFixed(2),
      fps: Math.round(1000 / frameMs),
      p90Ms: +times[Math.floor(times.length * 0.9)].toFixed(2),
      cpuMs: +median(samples.map((s) => s.cpu)).toFixed(2),
      drawnTriangles: stats.drawnTriangles,
      shadowTriangles: stats.shadowTriangles,
      drawnMeshlets: stats.drawnMeshlets,
      fullDetailTriangles: stats.fullDetailTriangles,
      instances: stats.instances,
      visibleInstances: stats.visibleInstances,
      roundsMs: rounds.map((r) => +r.toFixed(2)),
      /** Fraction of measured batches that ran with occlusion culling on (the automatic tuner decides). */
      occlusionShare: +(samples.filter((s) => s.occlusion).length / samples.length).toFixed(2),
    });
    console.log('BENCH pose', JSON.stringify(results[results.length - 1]));
  }

  const size = app.renderer.getDrawingBufferSize(new THREE.Vector2());
  const info = (device as unknown as { adapterInfo?: GPUAdapterInfo }).adapterInfo;
  const report = {
    scene: sceneName,
    mode,
    threshold,
    width: size.x,
    height: size.y,
    gpu: info ? [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') : 'unknown',
    userAgent: navigator.userAgent,
    results,
  };
  (window as unknown as { __bench: unknown }).__bench = report;
  console.log('BENCH', JSON.stringify(report));
  return report;
}
