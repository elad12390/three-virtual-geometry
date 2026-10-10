/**
 * Screen-space motion of VirtualMesh fragments, for temporal anti-aliasing (three's TRAA) and motion blur.
 *
 * three's `velocity` node reprojects the previous frame's position from the `position` geometry attribute, which a
 * VirtualMesh does not have (its vertices are pulled from storage buffers). This node reprojects the pulled
 * position instead, with the camera of the previous frame. Like three's node it uses the unjittered projection
 * that TRAA hands to `velocity`, so the jitter is not mistaken for motion. Instances are static, and vertex
 * deformation (wind) is not tracked: TAA's history clamping hides that small motion.
 *
 * VirtualGeometry swaps this output in automatically when a render pass asks for a `velocity` MRT output.
 */
import * as THREE from 'three/webgpu';
import { cameraViewMatrix, modelWorldMatrix, mrt, positionLocal, uniform, velocity, vec4 } from 'three/tsl';

type Node = any; // eslint-disable-line @typescript-eslint/no-explicit-any

interface CameraHistory {
  frame: number;
  projection: THREE.Matrix4;
  view: THREE.Matrix4;
  previousProjection: THREE.Matrix4;
  previousView: THREE.Matrix4;
}

class VirtualMeshVelocityNode extends THREE.TempNode {
  private readonly projection = uniform(new THREE.Matrix4());
  private readonly previousProjection = uniform(new THREE.Matrix4());
  private readonly previousView = uniform(new THREE.Matrix4());
  private readonly history = new WeakMap<THREE.Camera, CameraHistory>();

  constructor() {
    super('vec2');
    this.updateType = THREE.NodeUpdateType.RENDER;
  }

  update({ frameId, camera }: { frameId: number; camera: THREE.Camera | null }) {
    if (!camera) return undefined;
    // Set by TRAA while it renders: the projection without its jitter.
    const projection: THREE.Matrix4 = (velocity as Node).projectionMatrix ?? camera.projectionMatrix;
    let h = this.history.get(camera);
    if (!h) {
      h = { frame: frameId, projection: projection.clone(), view: camera.matrixWorldInverse.clone(), previousProjection: projection.clone(), previousView: camera.matrixWorldInverse.clone() };
      this.history.set(camera, h);
    } else if (h.frame !== frameId) {
      h.frame = frameId;
      h.previousProjection.copy(h.projection);
      h.previousView.copy(h.view);
      h.projection.copy(projection);
      h.view.copy(camera.matrixWorldInverse);
    }
    this.projection.value.copy(h.projection);
    this.previousProjection.value.copy(h.previousProjection);
    this.previousView.value.copy(h.previousView);
    return undefined;
  }

  setup() {
    const world: Node = (modelWorldMatrix as Node).mul(vec4(positionLocal as Node, 1));
    const current: Node = this.projection.mul(cameraViewMatrix as Node).mul(world);
    const previous: Node = this.previousProjection.mul(this.previousView).mul(world);
    return current.xy.div(current.w).sub(previous.xy.div(previous.w));
  }
}

/** Motion vector (NDC delta) of the current VirtualMesh fragment, in the format of three's `velocity`. */
export const vgVelocity: Node = new VirtualMeshVelocityNode();

/** MRT output that replaces a pass's `velocity` for VirtualMesh materials. */
export const vgVelocityOutput: Node = mrt({ velocity: vgVelocity });
