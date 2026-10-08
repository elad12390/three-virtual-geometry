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
export { buildVirtualMesh } from './core/preprocess/buildVirtualMesh';
export type { VirtualMeshBuildOptions, VirtualMeshData, VirtualMeshSource } from './core/preprocess/buildVirtualMesh';
export { partitionMeshlets } from './core/preprocess/partition';

export { buildVirtualMeshCached, clearVirtualMeshCache, virtualMeshCacheKey, VG_BUILD_VERSION } from './core/io/cache';
export type { VirtualGeometryCacheOptions, VirtualGeometryCachedBuildOptions } from './core/io/cache';
export { decodeVirtualMesh, encodeVirtualMesh, loadVirtualMesh, VirtualMeshFormatError, VG_FORMAT_VERSION } from './core/io/serialize';
export type { VirtualMeshEncodeOptions } from './core/io/serialize';

export { instanceGuaranteedRange, instanceMeshletRange, selectCut, verifyCutCoverage } from './core/runtime/cut';
export type { CutView } from './core/runtime/cut';
export { VirtualMesh, vgWorldNormal } from './core/runtime/VirtualMesh';
export { vgUv, vgTexture, vgNormalMap, bindVirtualGeometryTextures } from './core/runtime/vgMaterial';
export { toNodeMaterial, canConvertToNodeMaterial } from './core/import/toNodeMaterial';
export { virtualMeshesFromObject3D, collectVirtualGeometryGroups, VirtualGeometryImport } from './core/import/fromObject3D';
export type { VirtualGeometryImportOptions, VirtualGeometryImportGroup, VirtualGeometryBuilder } from './core/import/fromObject3D';
export type { VirtualMeshInstances, VirtualMeshOptions } from './core/runtime/VirtualMesh';
export { VirtualGeometry } from './core/runtime/VirtualGeometry';
export type { VirtualGeometryStats, VirtualGeometryOptions } from './core/runtime/VirtualGeometry';
export { VG_DEBUG_MODES, MAX_MESHLET_TRIANGLES, MAX_MESHLET_VERTICES } from './core/constants';

export { fromBufferGeometry, mergeSources } from './source';
export type { FromGeometryOptions } from './source';
