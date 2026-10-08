<script setup lang="ts">
import { computed } from 'vue';
import data from '../../benchmarks.json';

/** Uncapped FPS from docs/benchmarks.json (written by `npm run bench`). */
const props = withDefaults(defineProps<{ detail?: boolean }>(), { detail: false });

interface PoseResult {
  pose: string;
  frameMs: number;
  fps: number;
  p90Ms: number;
  cpuMs: number;
  drawnTriangles: number;
  shadowTriangles: number;
  fullDetailTriangles: number;
  instances: number;
  visibleInstances: number;
}
interface Run {
  scene: string;
  resolution: string;
  width: number;
  height: number;
  results: PoseResult[];
}

const LABELS: Record<string, string> = {
  ruins: 'Ruins',
  world: 'World (6 km)',
  map: 'Valley (2 km)',
  import: 'glTF import',
  stress: 'Stress, 1M instances',
};

const runs = data.runs as Run[];
const resolutions = [...new Set(runs.map((r) => r.resolution))];
const scenes = [...new Set(runs.map((r) => r.scene))];

const fmt = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : `${n}`;
const range = (xs: number[], f: (n: number) => string = (n) => n.toLocaleString('en-US')) => {
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  return lo === hi ? f(lo) : `${f(lo)} – ${f(hi)}`;
};
const run = (scene: string, resolution: string) => runs.find((r) => r.scene === scene && r.resolution === resolution);

const summary = computed(() =>
  scenes.map((scene) => {
    const all = runs.filter((r) => r.scene === scene).flatMap((r) => r.results);
    return {
      scene,
      label: LABELS[scene] ?? scene,
      full: fmt(Math.max(...all.map((r) => r.fullDetailTriangles))),
      instances: fmt(Math.max(...all.map((r) => r.instances))),
      drawn: range(all.map((r) => r.drawnTriangles), fmt),
      fps: resolutions.map((res) => {
        const r = run(scene, res);
        return r ? range(r.results.map((p) => p.fps)) : '–';
      }),
    };
  })
);

const details = computed(() =>
  scenes.map((scene) => {
    const first = run(scene, resolutions[0])!;
    return {
      scene,
      label: LABELS[scene] ?? scene,
      poses: first.results.map((p, i) => ({
        pose: p.pose,
        drawn: fmt(p.drawnTriangles),
        shadow: fmt(p.shadowTriangles),
        visible: fmt(p.visibleInstances),
        perRes: resolutions.map((res) => run(scene, res)?.results[i]),
      })),
    };
  })
);
</script>

<template>
  <div class="bench">
    <table>
      <thead>
        <tr>
          <th>Scene</th>
          <th>Full-detail triangles</th>
          <th>Instances</th>
          <th>Drawn per frame</th>
          <th v-for="res in resolutions" :key="res">FPS at {{ res }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in summary" :key="row.scene">
          <td>{{ row.label }}</td>
          <td>{{ row.full }}</td>
          <td>{{ row.instances }}</td>
          <td>{{ row.drawn }}</td>
          <td v-for="(fps, i) in row.fps" :key="i" class="fps">{{ fps }}</td>
        </tr>
      </tbody>
    </table>
    <p class="caption">
      Uncapped (no vsync): frames are submitted back to back and the GPU is waited on after every 8, so this is the
      sustained frame rate, not the display's. Each camera view is measured in 3 rounds of 96 frames (median round),
      after the automatic occlusion setting has settled; ranges span the views of each scene. Error threshold 1 px,
      shadows on (the stress test has none). {{ data.machine }}, {{ data.browser }}, {{ data.date }}.
    </p>

    <template v-if="props.detail">
      <details v-for="scene in details" :key="scene.scene">
        <summary>{{ scene.label }}: every camera view</summary>
        <table>
          <thead>
            <tr>
              <th>View</th>
              <th>Drawn</th>
              <th>Shadow</th>
              <th>Visible instances</th>
              <th v-for="res in resolutions" :key="res">{{ res }}: FPS (ms, CPU ms)</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="p in scene.poses" :key="p.pose">
              <td>{{ p.pose }}</td>
              <td>{{ p.drawn }}</td>
              <td>{{ p.shadow }}</td>
              <td>{{ p.visible }}</td>
              <td v-for="(r, i) in p.perRes" :key="i" class="fps">
                <template v-if="r">{{ r.fps.toLocaleString('en-US') }} <span class="ms">({{ r.frameMs }}, {{ r.cpuMs }})</span></template>
              </td>
            </tr>
          </tbody>
        </table>
      </details>
    </template>
  </div>
</template>

<style scoped>
.bench table {
  display: table;
  width: 100%;
}
.bench td {
  white-space: nowrap;
}
.fps {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  white-space: nowrap;
}
.ms {
  font-weight: 400;
  color: var(--vp-c-text-3);
}
.caption {
  font-size: 13px;
  color: var(--vp-c-text-2);
}
details {
  margin: 8px 0;
}
summary {
  cursor: pointer;
  font-weight: 600;
}
</style>
