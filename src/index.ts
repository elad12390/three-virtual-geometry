/**
 * three-virtual-geometry: Virtual geometry for three.js WebGPU.
 *
 * Quick start:
 *
 *   import { VirtualGeometry } from 'three-virtual-geometry';
 *
 *   const vg = new VirtualGeometry();   // once
 *   await vg.add(gltf.scene);         // converts its static meshes in place (materials, textures, instancing)
 *   scene.add(gltf.scene);
 *   renderer.render(scene, camera);       // LOD selection and culling run automatically before each render
 *
 * Lower level: `buildVirtualMeshCached(fromBufferGeometry(geometry))` once per mesh, then
 * `vg.createMesh(data, material, { matrices })` (16 floats per instance) and add it to a scene.
 */
export { buildVirtualMesh } from './core/preprocess/buildVirtualMesh.js';
export type { VirtualMeshBuildOptions, VirtualMeshData, VirtualMeshSource } from './core/preprocess/buildVirtualMesh.js';
export { partitionMeshlets } from './core/preprocess/partition.js';

export { buildVirtualMeshCached, clearVirtualMeshCache, virtualMeshCacheKey, VG_BUILD_VERSION } from './core/io/cache.js';
export type { VirtualGeometryCacheOptions, VirtualGeometryCachedBuildOptions } from './core/io/cache.js';
export { decodeVirtualMesh, encodeVirtualMesh, loadVirtualMesh, VirtualMeshFormatError, VG_FORMAT_VERSION } from './core/io/serialize.js';
export type { VirtualMeshEncodeOptions } from './core/io/serialize.js';

export { instanceGuaranteedRange, instanceMeshletRange, selectBlendCut, selectCut, verifyCutCoverage, verifyCutOverlap } from './core/runtime/cut.js';
export { lodFadeRange, LOD_FADE_STEPS } from './core/runtime/lodBlend.js';
export type { CutView } from './core/runtime/cut.js';
export { VirtualMesh, vgWorldNormal, vgInstanceOrigin } from './core/runtime/VirtualMesh.js';
export { vgUv, vgTexture, vgNormalMap, bindVirtualGeometryTextures } from './core/runtime/vgMaterial.js';
export { toNodeMaterial, canConvertToNodeMaterial } from './core/import/toNodeMaterial.js';
export { virtualMeshesFromObject3D, collectVirtualGeometryGroups, VirtualGeometryImport } from './core/import/fromObject3D.js';
export type { VirtualGeometryImportOptions, VirtualGeometryImportGroup, VirtualGeometryBuilder } from './core/import/fromObject3D.js';
export type { VirtualMeshInstances, VirtualMeshOptions } from './core/runtime/VirtualMesh.js';
export { VirtualGeometry } from './core/runtime/VirtualGeometry.js';
export type { VirtualGeometryStats, VirtualGeometryOptions } from './core/runtime/VirtualGeometry.js';
export { virtualGeometryLimits } from './core/runtime/limits.js';
export { VG_DEBUG_MODES, MAX_MESHLET_TRIANGLES, MAX_MESHLET_VERTICES } from './core/constants.js';

export { fromBufferGeometry, mergeSources } from './source.js';
export type { FromGeometryOptions } from './source.js';
