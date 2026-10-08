import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { VirtualGeometry, VG_DEBUG_MODES, virtualGeometryLimits, type VirtualGeometryStats } from '../../src/index';
import type { BenchPose } from './bench';

const HUD_INTERVAL_MS = 250;

const fmt = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : `${Math.round(n)}`;
/** Frame times kept for the HUD graph. */
const GRAPH_SAMPLES = 150;

const el = (id: string) => document.getElementById(id)!;

/** Renderer, camera, controls, HUD and debug GUI shared by the demo scenes. */
export class DemoApp {
  readonly renderer: THREE.WebGPURenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  readonly vg = new VirtualGeometry();
  readonly gui = new GUI({ title: 'Settings' });
  readonly settings = {
    triangleBudget: 8_000_000,
    // Quality first: a fixed 1 px threshold. The budget is a lever (it re-picks LODs whenever it adjusts).
    autoLod: false,
    debugView: 'shaded' as keyof typeof VG_DEBUG_MODES,
    frustumCulling: true,
    freezeLOD: false,
    uncapped: new URLSearchParams(location.search).has('uncapped'),
  };

  private readonly statsEl = el('stats');
  private readonly hud = {
    scene: el('hud-scene'),
    fps: el('hud-fps'),
    frame: el('hud-frame'),
    cpu: el('hud-cpu'),
    graph: el('hud-graph') as HTMLCanvasElement,
    full: el('hud-full'),
    drawn: el('hud-drawn'),
    ratio: el('hud-ratio'),
    budgetBar: el('hud-budget-bar'),
    budgetLabel: el('hud-budget-label'),
    budget: el('hud-budget'),
    instances: el('hud-instances'),
    visible: el('hud-visible'),
    visibleBar: el('hud-visible-bar'),
    meshlets: el('hud-meshlets'),
    threshold: el('hud-threshold'),
    occluded: el('hud-occluded'),
    shadow: el('hud-shadow'),
    warn: el('hud-warn'),
    extra: el('hud-extra'),
  };
  private readonly frameTimes = new Float32Array(GRAPH_SAMPLES);
  private frameTimeCursor = 0;
  private lastFrameTime = 0;
  private readonly overlayEl = document.getElementById('overlay')!;
  private readonly overlayText = document.getElementById('overlay-text')!;
  private readonly overlayBar = document.getElementById('overlay-bar')!;
  private lastStats: VirtualGeometryStats | null = null;
  /** HUD text is rebuilt and written to the DOM at most every HUD_INTERVAL_MS, not every frame. */
  private hudTime = -Infinity;
  private frames = 0;
  private fpsTime = performance.now();
  private fps = 0;
  private cpuMs = 0;
  extraStats = '';
  /** Scene name shown under the HUD title (HTML allowed). */
  sceneLabel = '';
  onFrame: ((dt: number) => void) | null = null;
  /** Camera views for `?bench`; scenes may set their own (default: the start view and an aerial one). */
  benchPoses: BenchPose[] | null = null;

  constructor() {
    this.renderer = new THREE.WebGPURenderer({ antialias: true, trackTimestamp: new URLSearchParams(location.search).has('timestamps') });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    document.body.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 5000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });

    this.gui.add(this.settings, 'triangleBudget', 250_000, 20_000_000, 250_000).name('triangle budget');
    this.gui.add(this.settings, 'autoLod').name('auto LOD (budget)');
    this.gui.add(this.vg.errorThreshold, 'value', 0.5, 64, 0.1).name('error threshold (px)').listen();
    this.gui.add(this.vg.minPixelRadius, 'value', 0, 8, 0.1).name('hide smaller than (px)').listen();
    this.gui.add(this.settings, 'debugView', Object.keys(VG_DEBUG_MODES) as (keyof typeof VG_DEBUG_MODES)[]).name('view');
    this.gui.add(this.settings, 'frustumCulling').name('frustum culling');
    this.gui.add(this.settings, 'freezeLOD').name('freeze LOD/culling');
    this.gui.add(this.settings, 'uncapped').name('uncapped fps (no vsync)').onChange(() => this.start());

    // Embedded in the docs (?embed): compact HUD, settings collapsed, and the page can pause rendering.
    if (new URLSearchParams(location.search).has('embed')) {
      document.body.classList.add('embed');
      this.gui.close();
    }
    window.addEventListener('message', (e) => {
      if (e.source !== window.parent || window.parent === window) return;
      if (e.data?.vg === 'pause') this.pause();
      else if (e.data?.vg === 'resume') this.resume();
    });
  }

  async init() {
    // Use the GPU's full buffer sizes, so single multi-million-triangle meshes (the forest's trees) fit.
    const limits = await virtualGeometryLimits();
    (this.renderer.backend as unknown as { parameters: { requiredLimits?: Record<string, number> } }).parameters.requiredLimits = limits;
    await this.renderer.init();
    if (!(this.renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend) {
      throw new Error('WebGPU is not available in this browser. This demo needs WebGPU (Chrome, Edge, Safari 26+).');
    }
  }

  progress(text: string, fraction: number) {
    this.overlayText.textContent = text;
    this.overlayBar.style.width = `${Math.round(fraction * 100)}%`;
  }

  hideOverlay() {
    this.overlayEl.style.display = 'none';
    this.statsEl.hidden = false;
    this.hud.scene.innerHTML = this.sceneLabel;
    // Controls hint: shown for a few seconds, gone for good after the first interaction.
    const hint = el('hint');
    hint.hidden = false;
    const fade = () => {
      hint.style.opacity = '0';
      setTimeout(() => (hint.hidden = true), 1300);
    };
    const timer = setTimeout(fade, 9000);
    this.renderer.domElement.addEventListener(
      'pointerdown',
      () => {
        clearTimeout(timer);
        fade();
      },
      { once: true }
    );
  }

  /**
   * Starts the render loop. Normally frames follow requestAnimationFrame, which the browser locks to
   * the display refresh rate. In uncapped mode frames are submitted back to back (MessageChannel ticks),
   * waiting for the GPU every few frames so the queue cannot run away: the screen still shows at most
   * the refresh rate, but the FPS counter reports how many frames the GPU can actually render.
   */
  start() {
    this.started = true;
    this.loopToken++;
    const token = this.loopToken;
    if (this.paused) {
      this.renderer.setAnimationLoop(null);
      return;
    }
    let last = performance.now();
    const step = () => {
      const now = performance.now();
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      this.renderFrame(dt);
      this.updateStats(now);
    };

    if (!this.settings.uncapped) {
      this.renderer.setAnimationLoop(step);
      return;
    }
    this.renderer.setAnimationLoop(null);
    const device = (this.renderer.backend as unknown as { device: GPUDevice }).device;
    const channel = new MessageChannel();
    let frames = 0;
    channel.port1.onmessage = async () => {
      if (token !== this.loopToken) return;
      step();
      if (++frames % 4 === 0) await device.queue.onSubmittedWorkDone();
      channel.port2.postMessage(null);
    };
    channel.port2.postMessage(null);
  }

  private loopToken = 0;
  private lastFrameId = -1;
  private started = false;
  private paused = false;

  /** Stops rendering (e.g. while an embedding page has the demo scrolled out of view). Loading continues. */
  pause() {
    this.paused = true;
    this.loopToken++;
    this.renderer.setAnimationLoop(null);
  }

  resume() {
    if (!this.paused) return;
    this.paused = false;
    this.lastFrameTime = 0;
    if (this.started) this.start();
  }

  /** One frame: settings -> VirtualGeometry LOD/culling on the GPU -> render. Used by the loop and the benchmark. */
  renderFrame(dt = 1 / 60) {
    const t0 = performance.now();
    // three.js advances its frame counter only inside setAnimationLoop, and per-frame work such as shadow maps runs
    // once per frame id. Frames driven by hand (uncapped mode, the benchmark) must advance it themselves, or shadows
    // are never redrawn and those frames come out cheaper than real ones.
    const nodeFrame = (this.renderer as unknown as { _nodes: { nodeFrame: { frameId: number; update(): void } } })._nodes.nodeFrame;
    if (nodeFrame.frameId === this.lastFrameId) {
      nodeFrame.update();
      (this.renderer.info as { frame: number }).frame = nodeFrame.frameId;
    }
    this.lastFrameId = nodeFrame.frameId;
    this.onFrame?.(dt);
    this.controls.update();
    const s = this.settings;
    this.vg.triangleBudget = s.autoLod ? s.triangleBudget : 0;
    const debugMode = VG_DEBUG_MODES[s.debugView];
    if (this.vg.debugMode.value !== debugMode) this.vg.debugMode.value = debugMode;
    const frustumCulling = s.frustumCulling ? 1 : 0;
    if (this.vg.frustumCulling.value !== frustumCulling) this.vg.frustumCulling.value = frustumCulling;
    this.vg.freeze = s.freezeLOD;
    if (!this.vg.autoUpdate) this.vg.update(this.renderer, this.camera);
    this.renderer.render(this.scene, this.camera); // runs the VirtualGeometry update first (autoUpdate)
    this.cpuMs = this.cpuMs * 0.9 + (performance.now() - t0) * 0.1;
  }

  private updateStats(now: number) {
    if (this.lastFrameTime > 0) {
      this.frameTimes[this.frameTimeCursor] = now - this.lastFrameTime;
      this.frameTimeCursor = (this.frameTimeCursor + 1) % GRAPH_SAMPLES;
    }
    this.lastFrameTime = now;
    this.frames++;
    if (now - this.fpsTime > 500) {
      this.fps = (this.frames * 1000) / (now - this.fpsTime);
      this.frames = 0;
      this.fpsTime = now;
    }
    // The context reads its stats back every few frames anyway (budget, capacity guard, occlusion).
    this.lastStats = this.vg.lastStats;
    if (now - this.hudTime < HUD_INTERVAL_MS) return;
    this.hudTime = now;
    this.renderHud();
  }

  private renderHud() {
    const h = this.hud;
    const s = this.lastStats;
    h.fps.textContent = this.fps.toFixed(0);
    h.fps.style.color = this.fps >= 55 ? '#f4f4f5' : this.fps >= 30 ? '#fbbf24' : '#f87171';
    h.frame.textContent = this.fps > 0 ? (1000 / this.fps).toFixed(1) : '--';
    h.cpu.textContent = this.cpuMs.toFixed(2);
    this.drawGraph();

    const threshold = this.vg.errorThreshold.value;
    h.threshold.innerHTML = `${threshold.toFixed(2)}<small>px</small>`;
    h.extra.textContent = this.extraStats;
    if (!s) return;
    h.full.textContent = fmt(s.fullDetailTriangles);
    h.drawn.textContent = fmt(s.drawnTriangles);
    const ratio = s.fullDetailTriangles / Math.max(1, s.drawnTriangles);
    h.ratio.textContent = `${ratio >= 100 ? Math.round(ratio).toLocaleString() : ratio.toFixed(1)}× fewer`;
    const budget = this.vg.triangleBudget;
    if (budget > 0) {
      h.budgetLabel.textContent = 'triangle budget';
      h.budget.textContent = `${fmt(s.drawnTriangles)} / ${fmt(budget)}`;
      h.budgetBar.style.width = `${Math.min(100, (100 * s.drawnTriangles) / budget)}%`;
    } else {
      h.budgetLabel.textContent = 'auto LOD off (fixed error)';
      h.budget.textContent = `${((100 * s.drawnTriangles) / Math.max(1, s.fullDetailTriangles)).toFixed(3)}% of scene`;
      h.budgetBar.style.width = '100%';
    }
    h.instances.textContent = fmt(s.instances);
    const visiblePct = (100 * s.visibleInstances) / Math.max(1, s.instances);
    h.visible.innerHTML = `${fmt(s.visibleInstances)}<small>${visiblePct.toFixed(visiblePct < 10 ? 1 : 0)}%</small>`;
    h.visibleBar.style.width = `${visiblePct}%`;
    h.meshlets.textContent = fmt(s.drawnMeshlets);
    const occ = this.vg.occlusion;
    h.occluded.textContent = occ.active ? fmt(s.occludedInstances) : occ.enabled && occ.auto ? 'auto: off' : 'off';
    h.shadow.textContent = fmt(s.shadowTriangles);
    h.warn.hidden = !s.overflow;
  }

  /** Frame-time history: filled area, 60 / 120 FPS guides, newest sample on the right. */
  private drawGraph() {
    const canvas = this.hud.graph;
    const dpr = Math.min(window.devicePixelRatio, 2);
    const w = Math.round(canvas.clientWidth * dpr);
    const hgt = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== hgt) {
      canvas.width = w;
      canvas.height = hgt;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx || w === 0) return;
    ctx.clearRect(0, 0, w, hgt);
    // Scale to the 95th percentile, so one hitch (or the loading frame) doesn't flatten the whole graph.
    const sorted = Array.from(this.frameTimes).sort((a, b) => a - b);
    const scale = Math.max(20, sorted[Math.floor(GRAPH_SAMPLES * 0.95)] * 1.3); // ms at the top of the graph
    const y = (ms: number) => hgt - (Math.min(ms, scale) / scale) * hgt;

    ctx.font = `${10 * dpr}px -apple-system, system-ui, sans-serif`;
    for (const [ms, label] of [
      [1000 / 60, '60'],
      [1000 / 120, '120'],
    ] as const) {
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.setLineDash([3 * dpr, 4 * dpr]);
      ctx.beginPath();
      ctx.moveTo(0, y(ms));
      ctx.lineTo(w, y(ms));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.fillText(label, w - ctx.measureText(label).width - 2 * dpr, y(ms) - 3 * dpr);
    }

    const step = w / (GRAPH_SAMPLES - 1);
    ctx.beginPath();
    for (let i = 0; i < GRAPH_SAMPLES; i++) {
      const t = this.frameTimes[(this.frameTimeCursor + i) % GRAPH_SAMPLES];
      if (i === 0) ctx.moveTo(0, y(t));
      else ctx.lineTo(i * step, y(t));
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 1.5 * dpr;
    ctx.stroke();
    ctx.lineTo(w, hgt);
    ctx.lineTo(0, hgt);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,255,255,0.06)';
    ctx.fill();
  }
}
