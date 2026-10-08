// Entry of the all-in-one CDN build (dist/three-virtual-geometry.all.min.js): the library plus three.js (WebGPU build
// and TSL), meshoptimizer and the addons most pages need, in one file. One import, no import map.
export * from '../../src/index';
export * as THREE from 'three/webgpu';
export * as TSL from 'three/tsl';
export { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
export { OrbitControls } from 'three/addons/controls/OrbitControls.js';
export { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
