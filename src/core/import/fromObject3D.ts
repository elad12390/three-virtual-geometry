/**
 * One-call scene import: every static mesh of an Object3D (e.g. a loaded glTF scene) becomes VirtualGeometry geometry.
 *
 * Meshes are grouped by (geometry, geometry group, material, render flags), so repeated geometry becomes one
 * VirtualMesh with one instance per occurrence (InstancedMesh instances included). Each unique geometry range
 * is built once. Skinned and morphing meshes, points, lines and sprites are left alone and keep rendering.
 */
import * as THREE from 'three/webgpu';
import { buildVirtualMeshCached, type VirtualGeometryCachedBuildOptions } from '../io/cache.js';
import { buildVirtualMesh, type VirtualMeshBuildOptions, type VirtualMeshData, type VirtualMeshSource } from '../preprocess/buildVirtualMesh.js';
import { fromBufferGeometry } from '../../source.js';
import type { VirtualGeometry } from '../runtime/VirtualGeometry.js';
import type { VirtualMesh, VirtualMeshOptions } from '../runtime/VirtualMesh.js';
import { canConvertToNodeMaterial, toNodeMaterial } from './toNodeMaterial.js';

/** Builds one DAG. Same signature as `buildVirtualMesh`, so a caching builder drops in here or via `options.builder`. */
export type VirtualGeometryBuilder = (source: VirtualMeshSource, options?: VirtualMeshBuildOptions) => Promise<VirtualMeshData>;
// Cached in IndexedDB: importing the same model again (e.g. the next page load) skips the build.
const defaultBuilder: VirtualGeometryBuilder = buildVirtualMeshCached;

export interface VirtualGeometryImportOptions {
  /**
   * DAG build options for every geometry (`onProgress` is driven by the import; use `onProgress` below), including
   * `cache` options for the default cached builder, e.g. `{ cache: { maxBytes: 2 * 1024 ** 3 } }`.
   */
  build?: Omit<VirtualGeometryCachedBuildOptions, 'onProgress'>;
  /** Options for each created VirtualMesh (e.g. `maxDrawDistance`), or a function returning them per group. */
  mesh?: VirtualMeshOptions | ((group: VirtualGeometryImportGroup) => VirtualMeshOptions | undefined);
  /** Return false to leave a mesh as it is (it keeps rendering normally). Called for every compatible mesh. */
  filter?: (mesh: THREE.Mesh) => boolean;
  /**
   * true (default): the VirtualMeshes are added to `object` and the converted meshes stop rendering (their
   * leaf meshes are taken out of the graph, meshes with children stay with their layers cleared; `dispose()`
   * restores both). false: nothing in the scene changes; add `result.object` yourself.
   */
  replace?: boolean;
  /** Overall progress in 0..1 while building. May be async (awaited), e.g. to let the page repaint. */
  onProgress?: (fraction: number) => void | Promise<void>;
  /** Converts a source material. Default: node materials are cloned, classic mesh materials go through `toNodeMaterial`. */
  material?: (material: THREE.Material, group: VirtualGeometryImportGroup) => THREE.NodeMaterial | null;
  /** DAG builder (default `buildVirtualMesh`), e.g. a cached one with the same signature. */
  builder?: VirtualGeometryBuilder;
}

/** Meshes that share geometry, geometry range, material and render flags: one VirtualMesh. */
export interface VirtualGeometryImportGroup {
  geometry: THREE.BufferGeometry;
  /** Index range (vertex range for non-indexed geometry) this group draws: one entry of `geometry.groups`, clipped to the draw range. */
  range: { start: number; count: number };
  material: THREE.Material;
  /** Column-major world matrix per instance (16 floats each), in the order of `instances`. */
  matrices: Float32Array;
  /** rgba per instance, from `InstancedMesh.instanceColor` (null when no source has instance colors). */
  colors: Float32Array | null;
  /** Source of each instance: the mesh, and the instance index for an InstancedMesh (else -1). */
  instances: { object: THREE.Mesh; index: number }[];
  /** Per-vertex colors are used (`material.vertexColors` and a `color` attribute), like three would. */
  vertexColors: boolean;
  castShadow: boolean;
  receiveShadow: boolean;
  /** False when the mesh or one of its ancestors is hidden. */
  visible: boolean;
  renderOrder: number;
  layers: number;
  /** From the first source mesh. */
  name: string;
  userData: Record<string, unknown>;
}

const converted = new WeakSet<THREE.Object3D>();

/** Why a mesh is left alone, or null when it can be converted. */
export function incompatibility(mesh: THREE.Object3D): string | null {
  const m = mesh as THREE.Mesh & { isSkinnedMesh?: boolean; isBatchedMesh?: boolean; isVirtualMesh?: boolean; isVirtualMeshPart?: boolean };
  if (!m.isMesh) return 'not a mesh';
  if (m.isVirtualMesh || m.isVirtualMeshPart) return 'already a VirtualMesh';
  if (m.isSkinnedMesh) return 'skinned';
  if (m.isBatchedMesh) return 'batched';
  const geometry = m.geometry;
  if (!geometry?.attributes?.position || (geometry as THREE.InstancedBufferGeometry).isInstancedBufferGeometry) return 'unsupported geometry';
  const morph = geometry.morphAttributes;
  if ((morph.position?.length ?? 0) > 0 || (morph.normal?.length ?? 0) > 0 || (m.morphTargetInfluences?.length ?? 0) > 0) return 'morph targets';
  const materials = Array.isArray(m.material) ? m.material : [m.material];
  if (materials.some((mat) => !mat || !canConvertToNodeMaterial(mat))) return 'material without a node equivalent';
  return null;
}

/**
 * Collects the convertible meshes under `root` into instance groups (pure CPU work, no GPU, testable in Node).
 * World matrices are read after updating them, so the result matches what three would render now.
 */
export function collectVirtualGeometryGroups(root: THREE.Object3D, filter?: (mesh: THREE.Mesh) => boolean) {
  root.updateWorldMatrix(true, true);
  const groups = new Map<string, VirtualGeometryImportGroup & { matrixList: number[]; colorList: number[] | null }>();
  const meshes: THREE.Mesh[] = [];
  const skipped: { object: THREE.Object3D; reason: string }[] = [];
  const instanceMatrix = new THREE.Matrix4();
  const world = new THREE.Matrix4();
  const color = new THREE.Color();

  root.traverse((object) => {
    if (!(object as THREE.Mesh).isMesh || converted.has(object)) return;
    const mesh = object as THREE.Mesh;
    const reason = incompatibility(mesh) ?? (filter && !filter(mesh) ? 'filtered out' : null);
    if (reason) {
      skipped.push({ object: mesh, reason });
      return;
    }
    const geometry = mesh.geometry;
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    const drawn = drawnRanges(geometry, Array.isArray(mesh.material));
    if (drawn.length === 0) return;
    meshes.push(mesh);
    const instanced = (mesh as THREE.InstancedMesh).isInstancedMesh ? (mesh as THREE.InstancedMesh) : null;
    const count = instanced ? instanced.count : 1;
    const visible = isVisible(mesh);

    for (const { start, count: rangeCount, materialIndex } of drawn) {
      const material = materials[materialIndex];
      if (!material) continue;
      const vertexColors = material.vertexColors === true && geometry.attributes.color !== undefined;
      const key = [geometry.id, start, rangeCount, material.uuid, mesh.castShadow, mesh.receiveShadow, visible, mesh.renderOrder, mesh.layers.mask].join(':');
      let group = groups.get(key);
      if (!group) {
        group = {
          geometry,
          range: { start, count: rangeCount },
          material,
          matrices: new Float32Array(0),
          colors: null,
          instances: [],
          vertexColors,
          castShadow: mesh.castShadow,
          receiveShadow: mesh.receiveShadow,
          visible,
          renderOrder: mesh.renderOrder,
          layers: mesh.layers.mask,
          name: mesh.name,
          userData: { ...mesh.userData },
          matrixList: [],
          colorList: null,
        };
        groups.set(key, group);
      }
      for (let i = 0; i < count; i++) {
        if (instanced) {
          instanced.getMatrixAt(i, instanceMatrix);
          world.multiplyMatrices(mesh.matrixWorld, instanceMatrix);
        } else world.copy(mesh.matrixWorld);
        group.matrixList.push(...world.elements);
        const tinted = !!instanced?.instanceColor;
        if (tinted) instanced!.getColorAt(i, color);
        if (tinted && !group.colorList) group.colorList = new Array(group.instances.length * 4).fill(1);
        if (group.colorList) {
          if (tinted) group.colorList.push(color.r, color.g, color.b, 1);
          else group.colorList.push(1, 1, 1, 1);
        }
        group.instances.push({ object: mesh, index: instanced ? i : -1 });
      }
    }
  });

  const result: VirtualGeometryImportGroup[] = [];
  for (const { matrixList, colorList, ...group } of groups.values()) {
    group.matrices = Float32Array.from(matrixList);
    group.colors = colorList ? Float32Array.from(colorList) : null;
    result.push(group);
  }
  return { groups: result, meshes, skipped };
}

/** The (start, count, materialIndex) ranges three draws for this geometry, clipped to its draw range. */
function drawnRanges(geometry: THREE.BufferGeometry, multiMaterial: boolean) {
  const total = geometry.index ? geometry.index.count : geometry.attributes.position.count;
  const drawStart = Math.max(0, geometry.drawRange.start);
  const drawEnd = Math.min(total, drawStart + geometry.drawRange.count);
  const ranges = multiMaterial ? geometry.groups : [{ start: 0, count: total, materialIndex: 0 }];
  const out: { start: number; count: number; materialIndex: number }[] = [];
  for (const g of ranges) {
    const start = Math.max(g.start, drawStart);
    const end = Math.min(g.start + g.count, drawEnd);
    if (end - start >= 3) out.push({ start, count: end - start, materialIndex: g.materialIndex ?? 0 });
  }
  return out;
}

function isVisible(object: THREE.Object3D) {
  for (let o: THREE.Object3D | null = object; o; o = o.parent) if (!o.visible) return false;
  return true;
}

/** Result of `VirtualGeometry.add()`: the created meshes, and how to undo the import. */
export class VirtualGeometryImport {
  /** Holds every created VirtualMesh. Added to the imported object unless `replace: false`. */
  readonly object = new THREE.Group();
  /** One VirtualMesh per entry of `groups`. */
  readonly meshes: VirtualMesh[] = [];
  groups: VirtualGeometryImportGroup[];
  /** Meshes that were left alone, and why. */
  readonly skipped: { object: THREE.Object3D; reason: string }[];
  readonly stats = { sourceMeshes: 0, instances: 0, uniqueGeometries: 0, buildMs: 0 };
  private readonly hidden = new Map<THREE.Object3D, number>();
  /** Converted leaf meshes taken out of the graph, with their former parent (restored by `dispose`). */
  private readonly detached = new Map<THREE.Object3D, THREE.Object3D>();

  constructor(
    private readonly context: VirtualGeometry,
    collected: ReturnType<typeof collectVirtualGeometryGroups>
  ) {
    this.object.name = 'VirtualGeometryImport';
    this.groups = collected.groups;
    this.skipped = collected.skipped;
    this.stats.sourceMeshes = collected.meshes.length;
    this.stats.instances = collected.groups.reduce((a, g) => a + g.instances.length, 0);
  }

  /** @internal Stops the given source meshes from rendering (restored by `dispose`). */
  hide(meshes: THREE.Mesh[], root: THREE.Object3D) {
    const emptied = new Set<THREE.Object3D>();
    for (const mesh of meshes) {
      converted.add(mesh);
      // Leaf meshes leave the graph: three.js would still traverse them every frame (matrix updates and
      // render-list projection), which costs more CPU than the rendering itself in scenes with thousands of
      // props. Meshes with children (lights, skinned parts, ...) stay, with their layers cleared.
      if (mesh.children.length === 0 && mesh.parent) {
        this.detached.set(mesh, mesh.parent);
        emptied.add(mesh.parent);
        mesh.parent.remove(mesh);
        continue;
      }
      if (!this.hidden.has(mesh)) this.hidden.set(mesh, mesh.layers.mask);
      mesh.layers.disableAll();
    }
    // Plain groups left empty (e.g. one root node per prop) go too, walking up, but never `root` itself.
    for (let node of emptied) {
      while (node !== root && node.children.length === 0 && node.parent && (node.type === 'Group' || node.type === 'Object3D')) {
        const parent: THREE.Object3D = node.parent;
        this.detached.set(node, parent);
        parent.remove(node);
        node = parent;
      }
    }
  }

  /** World matrix of a source object, also when it (or its ancestors) were taken out of the graph. */
  private worldMatrixOf(object: THREE.Object3D): THREE.Matrix4 {
    const parent = this.detached.get(object);
    if (!parent) {
      object.updateWorldMatrix(true, false);
      return object.matrixWorld;
    }
    object.updateMatrix();
    return object.matrixWorld.multiplyMatrices(this.worldMatrixOf(parent), object.matrix);
  }

  /**
   * Re-reads every source mesh's current world matrix into the instances (call after moving the imported
   * object or animating its nodes). Instances are baked in world space at import time otherwise.
   */
  syncTransforms() {
    const m = new THREE.Matrix4();
    const instanceMatrix = new THREE.Matrix4();
    const roots = new Set<THREE.Object3D>();
    this.groups.forEach((group, g) => {
      const mesh = this.meshes[g];
      group.instances.forEach(({ object, index }, i) => {
        if (!roots.has(object)) {
          this.worldMatrixOf(object);
          roots.add(object);
        }
        if (index >= 0) {
          (object as THREE.InstancedMesh).getMatrixAt(index, instanceMatrix);
          m.multiplyMatrices(object.matrixWorld, instanceMatrix);
        } else m.copy(object.matrixWorld);
        mesh.setMatrixAt(i, m);
      });
      mesh.commitInstances();
    });
  }

  /** Removes the VirtualMeshes and lets the source meshes render again. */
  dispose() {
    for (const mesh of this.meshes) {
      mesh.dispose(this.context);
      mesh.material.dispose();
      mesh.geometry.dispose();
    }
    this.meshes.length = 0;
    for (const [mesh, mask] of this.hidden) {
      mesh.layers.mask = mask;
      converted.delete(mesh);
    }
    this.hidden.clear();
    for (const [mesh, parent] of this.detached) {
      parent.add(mesh);
      converted.delete(mesh);
    }
    this.detached.clear();
    this.object.removeFromParent();
  }
}

/**
 * Converts every compatible mesh under `object` to VirtualGeometry geometry. See `VirtualGeometry.add`.
 */
export async function virtualMeshesFromObject3D(context: VirtualGeometry, object: THREE.Object3D, options: VirtualGeometryImportOptions = {}): Promise<VirtualGeometryImport> {
  const t0 = performance.now();
  const build = options.builder ?? defaultBuilder;
  const collected = collectVirtualGeometryGroups(object, options.filter);
  const result = new VirtualGeometryImport(context, collected);

  // One build per geometry range (and vertex color use): groups that differ only in material or flags share it.
  const sources = new Map<string, { group: VirtualGeometryImportGroup; triangles: number; data?: VirtualMeshData }>();
  const sourceKey = (g: VirtualGeometryImportGroup) => `${g.geometry.id}:${g.range.start}:${g.range.count}:${g.vertexColors}`;
  for (const group of result.groups) {
    const key = sourceKey(group);
    if (!sources.has(key)) sources.set(key, { group, triangles: group.range.count / 3 });
  }
  result.stats.uniqueGeometries = sources.size;
  const total = [...sources.values()].reduce((a, s) => a + s.triangles, 0) || 1;
  let done = 0;
  for (const source of sources.values()) {
    const { geometry, range, vertexColors } = source.group;
    const mesh = fromBufferGeometry(geometry, {
      range,
      colors: vertexColors,
      normals: geometry.attributes.normal ? 'keep' : 'compute',
    });
    source.data = await build(mesh, {
      ...options.build,
      onProgress: (f) => options.onProgress?.((done + f * source.triangles) / total),
    });
    done += source.triangles;
  }

  const materials = new Map<string, THREE.NodeMaterial | null>();
  const createdGroups: VirtualGeometryImportGroup[] = [];
  const failed = new Set<THREE.Object3D>();
  for (const group of result.groups) {
    const material = convertMaterial(group, options, materials);
    if (!material) {
      for (const { object: source } of group.instances) failed.add(source);
      continue;
    }
    const meshOptions = typeof options.mesh === 'function' ? options.mesh(group) : options.mesh;
    const mesh = context.createMesh(sources.get(sourceKey(group))!.data!, material, { matrices: group.matrices, colors: group.colors ?? undefined }, meshOptions);
    // Vertices are pulled in world space: the mesh itself must stay at identity wherever it is attached.
    mesh.matrixAutoUpdate = false;
    mesh.matrixWorldAutoUpdate = false;
    mesh.name = group.name;
    mesh.userData = { ...group.userData };
    mesh.castShadow = group.castShadow;
    mesh.receiveShadow = group.receiveShadow;
    mesh.visible = group.visible;
    mesh.renderOrder = group.renderOrder;
    mesh.layers.mask = group.layers;
    result.meshes.push(mesh);
    createdGroups.push(group);
    result.object.add(mesh);
  }
  result.groups = createdGroups;

  if (options.replace ?? true) {
    // A mesh with any part that could not be converted keeps rendering (its converted parts are drawn twice).
    result.hide(collected.meshes.filter((m) => !failed.has(m)), object);
    object.add(result.object);
  }
  await options.onProgress?.(1);
  result.stats.buildMs = performance.now() - t0;
  return result;
}

/**
 * Node material for a group. Node materials are cloned so the source mesh keeps its own (VirtualMesh rewires the
 * material it gets). glTF normal maps authored against tangents need the derivative-tangent sign (normalScale.y
 * flipped), which GLTFLoader only applies when the geometry has no tangents.
 */
function convertMaterial(group: VirtualGeometryImportGroup, options: VirtualGeometryImportOptions, cache: Map<string, THREE.NodeMaterial | null>) {
  const source = group.material as THREE.MeshStandardMaterial;
  const flipNormalY = !!(source.normalMap && group.geometry.attributes.tangent && source.normalMap.flipY === false);
  const key = `${source.uuid}:${flipNormalY}`;
  if (cache.has(key)) return cache.get(key)!;
  let material: THREE.NodeMaterial | null;
  if (options.material) material = options.material(source, group);
  else {
    material = toNodeMaterial(source);
    if (material === (source as THREE.Material)) material = (source as unknown as THREE.NodeMaterial).clone();
  }
  if (material && flipNormalY) {
    const scale = (material as unknown as { normalScale?: THREE.Vector2 }).normalScale;
    if (scale) (material as unknown as { normalScale: THREE.Vector2 }).normalScale = scale.clone().setY(-scale.y);
  }
  cache.set(key, material);
  return material;
}
