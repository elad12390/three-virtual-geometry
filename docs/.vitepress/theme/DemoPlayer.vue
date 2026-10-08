<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { withBase } from 'vitepress';

/**
 * All demo scenes in one place. Only the selected scene runs: switching replaces the iframe, which frees the previous
 * scene's GPU device, and rendering pauses while the player is scrolled out of view.
 */
const props = withDefaults(defineProps<{ autoplay?: boolean; initial?: string }>(), { autoplay: false, initial: 'ruins' });

interface DemoScene {
  key: string;
  name: string;
  stat: string;
  query: string;
  poster: string;
  description: string;
}

const scenes: DemoScene[] = [
  {
    key: 'ruins',
    name: 'Ruins',
    stat: '5.7B triangles',
    query: 'scene=ruins&tour',
    poster: '/screenshots/ruins-wide.jpg',
    description:
      'An ancient city where every brick, paving stone, column flute and carving is geometry: 113k instances of 23 assets, drawn with about 5M triangles a frame. The camera flies from a 2-million-triangle bronze out to the whole site; drag to take over.',
  },
  {
    key: 'world',
    name: 'World',
    stat: '3M instances',
    query: 'scene=world',
    poster: '/screenshots/world.jpg',
    description:
      'A 6 km landscape with forests, rocks, grass and flowers: about 3 million instances and 37.5 billion full-detail triangles. Try the "meshlets" view in the settings panel.',
  },
  {
    key: 'map',
    name: 'Valley',
    stat: '2 km map',
    query: 'scene=map',
    poster: '/screenshots/map.jpg',
    description: 'A 2 km valley with conifers, a lake and mountains, generated at load time and lit with shadows.',
  },
  {
    key: 'import',
    name: 'glTF import',
    stat: '12k meshes → 58',
    query: 'scene=import',
    poster: '/screenshots/import.jpg',
    description:
      'A parking lot of textured glTF cars (Kenney Car Kit, CC0). 12,366 meshes become 58 instanced virtual meshes with one vg.add() call, textures included.',
  },
  {
    key: 'stress',
    name: 'Stress test',
    stat: '1M instances',
    query: 'scene=stress&count=1000000',
    poster: '/screenshots/stress.jpg',
    description: 'A million instances on a flat field, 39 billion full-detail triangles, culled in cells of 128 on the GPU.',
  },
];

// In `docs:dev` the demo runs on its own Vite server (npm run dev); in the build it is copied to /demo/.
const demoBase = import.meta.env.DEV ? 'http://localhost:8090/' : withBase('/demo/');

const selected = ref(scenes.find((s) => s.key === props.initial)?.key ?? scenes[0].key);
const scene = computed(() => scenes.find((s) => s.key === selected.value)!);
const running = ref(false);
const webgpu = ref(true);
const stage = ref<HTMLElement | null>(null);
const frame = ref<HTMLIFrameElement | null>(null);
const src = computed(() => `${demoBase}?${scene.value.query}&embed`);
const standalone = computed(() => `${demoBase}?${scene.value.query}`);

let visible = true;
let observer: IntersectionObserver | null = null;

function post(message: 'pause' | 'resume') {
  frame.value?.contentWindow?.postMessage({ vg: message }, '*');
}

function select(key: string) {
  selected.value = key;
  if (typeof history !== 'undefined') history.replaceState(null, '', `#${key}`);
}

function run() {
  if (webgpu.value) running.value = true;
}

function fullscreen() {
  stage.value?.requestFullscreen?.();
}

// A fresh iframe starts paused if the player is off screen.
function onFrameLoad() {
  if (!visible) post('pause');
}

onMounted(() => {
  webgpu.value = 'gpu' in navigator;
  const fromHash = location.hash.slice(1);
  if (scenes.some((s) => s.key === fromHash)) selected.value = fromHash;
  observer = new IntersectionObserver(
    ([entry]) => {
      visible = entry.isIntersecting;
      post(visible ? 'resume' : 'pause');
      if (visible && props.autoplay) run();
    },
    { threshold: 0.15 }
  );
  if (stage.value) observer.observe(stage.value);
});

onBeforeUnmount(() => observer?.disconnect());
</script>

<template>
  <div class="demo-player">
    <div class="tabs" role="tablist" aria-label="Demo scenes">
      <button
        v-for="s in scenes"
        :key="s.key"
        role="tab"
        :aria-selected="s.key === selected"
        :class="{ active: s.key === selected }"
        @click="select(s.key)"
      >
        <span class="name">{{ s.name }}</span>
        <span class="stat">{{ s.stat }}</span>
      </button>
    </div>

    <div ref="stage" class="stage">
      <iframe
        v-if="running"
        ref="frame"
        :key="src"
        :src="src"
        :title="`${scene.name} demo`"
        allow="fullscreen"
        @load="onFrameLoad"
      />
      <div v-else class="poster">
        <img :src="withBase(scene.poster)" :alt="scene.name" />
        <div class="poster-body">
          <template v-if="webgpu">
            <button class="run" @click="run">Run {{ scene.name }} live</button>
            <span class="note">Runs in this page with WebGPU. The first load builds the scene, then it is cached.</span>
          </template>
          <span v-else class="note">
            This browser has no WebGPU, so the scene can't run here. Try Chrome, Edge or Safari 26+.
          </span>
        </div>
      </div>
    </div>

    <div class="meta">
      <p>{{ scene.description }}</p>
      <div class="actions">
        <button v-if="running" @click="fullscreen">Fullscreen</button>
        <a :href="standalone" target="_blank" rel="noopener">Open in a new tab ↗</a>
      </div>
    </div>
  </div>
</template>

<style scoped>
.demo-player {
  margin: 24px 0;
}
.tabs {
  display: flex;
  gap: 6px;
  overflow-x: auto;
  padding-bottom: 8px;
  scrollbar-width: thin;
}
.tabs button {
  flex: 0 0 auto;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  padding: 8px 14px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: border-color 0.15s, color 0.15s, background 0.15s;
}
.tabs button:hover {
  color: var(--vp-c-text-1);
  border-color: var(--vp-c-text-3);
}
.tabs button.active {
  color: var(--vp-c-text-1);
  border-color: var(--vp-c-text-1);
  background: var(--vp-c-bg-soft);
}
.tabs .name {
  font-weight: 600;
  font-size: 14px;
}
.tabs .stat {
  font-size: 12px;
  color: var(--vp-c-text-3);
}
.stage {
  position: relative;
  aspect-ratio: 16 / 9;
  max-height: calc(100vh - 200px);
  width: 100%;
  border-radius: 10px;
  overflow: hidden;
  background: #0c0c0d;
}
.stage:fullscreen {
  max-height: none;
  border-radius: 0;
}
.stage iframe {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  border: 0;
}
.poster img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0.55;
}
.poster-body {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 24px;
  text-align: center;
}
.run {
  padding: 12px 22px;
  border-radius: 999px;
  border: 0;
  background: #f4f1ea;
  color: #121211;
  font-weight: 600;
  font-size: 15px;
  cursor: pointer;
}
.run:hover {
  background: #fff;
}
.note {
  max-width: 440px;
  font-size: 13px;
  color: #e8e4da;
  text-shadow: 0 1px 6px rgba(0, 0, 0, 0.8);
}
.meta {
  display: flex;
  gap: 24px;
  align-items: flex-start;
  justify-content: space-between;
  margin-top: 12px;
}
.meta p {
  margin: 0;
  font-size: 14px;
  line-height: 1.6;
  color: var(--vp-c-text-2);
}
.actions {
  display: flex;
  gap: 14px;
  flex: 0 0 auto;
  align-items: center;
  font-size: 14px;
}
.actions button {
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  padding: 4px 10px;
  color: var(--vp-c-text-1);
  cursor: pointer;
}
.actions a {
  color: var(--vp-c-brand-1);
  text-decoration: none;
}
@media (max-width: 640px) {
  .meta {
    flex-direction: column;
    gap: 8px;
  }
}
</style>
