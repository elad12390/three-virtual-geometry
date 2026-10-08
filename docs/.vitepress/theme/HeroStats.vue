<script setup lang="ts">
import { withBase } from 'vitepress';
import data from '../../benchmarks.json';

/** Headline results under the home page hero, computed from docs/benchmarks.json (`npm run bench`). */
interface Result {
  fps: number;
  cpuMs: number;
  fullDetailTriangles: number;
  instances: number;
}
interface Run {
  scene: string;
  resolution: string;
  results: Result[];
}
const runs = data.runs as Run[];

const of = (scene: string, resolution: string) => runs.find((r) => r.scene === scene && r.resolution === resolution)?.results ?? [];
const fpsRange = (scene: string, resolution: string) => {
  const fps = of(scene, resolution).map((r) => r.fps);
  if (!fps.length) return '–';
  const lo = Math.min(...fps);
  const hi = Math.max(...fps);
  return lo === hi ? `${lo}` : `${lo}–${hi.toLocaleString('en-US')}`;
};
const billions = (scene: string) => (Math.max(...of(scene, '1080p').map((r) => r.fullDetailTriangles)) / 1e9).toFixed(1);
const millions = (scene: string) => Math.round(Math.max(...of(scene, '1080p').map((r) => r.instances)) / 1e6);
const large = runs.filter((r) => r.scene !== 'import').flatMap((r) => r.results);
const cpu = Math.max(...large.map((r) => r.cpuMs)).toFixed(1);
const minFps4k = Math.min(...runs.filter((r) => r.resolution === '4K').flatMap((r) => r.results.map((p) => p.fps)));

const cards = [
  {
    value: `${billions('ruins')} billion`,
    label: 'triangles in the ruins scene',
    detail: `${fpsRange('ruins', '1080p')} FPS at 1080p · ${fpsRange('ruins', '4K')} at 4K`,
  },
  {
    value: `${billions('world')} billion`,
    label: `triangles, ${millions('world')} million instances (6 km world)`,
    detail: `${fpsRange('world', '1080p')} FPS at 1080p · ${fpsRange('world', '4K')} at 4K`,
  },
  {
    value: `${millions('stress')} million`,
    label: 'instances in the stress test',
    detail: `${fpsRange('stress', '1080p')} FPS at 1080p`,
  },
  {
    value: `${minFps4k}+ FPS`,
    label: 'in every scene and view, even at 4K',
    detail: `at most ${cpu} ms of CPU time per frame`,
  },
];
</script>

<template>
  <section class="hero-stats" aria-label="Benchmark results">
    <div class="inner">
    <div class="cards">
      <div v-for="c in cards" :key="c.label" class="card">
        <div class="value">{{ c.value }}</div>
        <div class="label">{{ c.label }}</div>
        <div class="detail">{{ c.detail }}</div>
      </div>
    </div>
    <p class="note">
      Measured uncapped (no vsync), shadows on: {{ data.machine.split(',')[0] }}, {{ data.browser.split('.')[0] }}. <a :href="withBase('/guide/performance#benchmarks')">Every scene and camera view</a> ·
      <a :href="withBase('/live')">run them yourself</a>
    </p>
    </div>
  </section>
</template>

<style scoped>
.hero-stats {
  padding: 8px 24px 40px;
}
@media (min-width: 640px) {
  .hero-stats {
    padding: 8px 48px 40px;
  }
}
@media (min-width: 960px) {
  .hero-stats {
    padding: 0 64px 48px;
  }
}
/* Same width as the hero and the feature cards. */
.inner {
  max-width: 1152px;
  margin: 0 auto;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}
.card {
  padding: 20px 22px;
  border-radius: 10px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
}
.value {
  font-size: 30px;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.15;
  color: var(--vp-c-text-1);
  font-variant-numeric: tabular-nums;
}
.label {
  margin-top: 4px;
  font-size: 14px;
  color: var(--vp-c-text-2);
}
.detail {
  margin-top: 10px;
  font-size: 14px;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  font-variant-numeric: tabular-nums;
}
.note {
  margin-top: 14px;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.note a {
  color: var(--vp-c-brand-1);
}
</style>
