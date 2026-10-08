/**
 * Texture coordinates for materials on a VirtualMesh.
 *
 * A VirtualMesh pulls its vertices from storage buffers, so its placeholder geometry has no `uv` attribute and
 * three's default texture lookups (`uv()`) would read nothing. The vertex stage writes the pulled UV into the
 * `vgUv` varying, and `bindVirtualGeometryTextures` points the material's textures at it.
 */
import * as THREE from 'three/webgpu';
import {
  bumpMap,
  context,
  float,
  mat3,
  materialColor,
  materialReference,
  negateOnBackSide,
  normalView,
  positionView,
  sqrt,
  texture,
  transformNormalToView,
  varyingProperty,
  vec3,
  vec4,
} from 'three/tsl';

type Node = any; // eslint-disable-line @typescript-eslint/no-explicit-any

/** Interpolated texture coordinates of the current fragment (vec2), for custom material nodes. */
export const vgUv = varyingProperty('vec2', 'vVgUv');

/** Plain 2D textures sample `vgUv`; cube, 3D and environment lookups keep their own coordinates. */
function getUV(node: Node): Node {
  if (node?.isTextureNode !== true || node.isCubeTextureNode || node.isTexture3DNode) return undefined;
  return vgUv;
}

/** Texture lookup at `vgUv`, with the texture's offset/repeat/rotation applied like three's own maps. */
export function vgTexture(map: THREE.Texture): Node {
  return (texture(map, vgUv) as Node).setUpdateMatrix(true);
}

const TWO_CHANNEL_FORMATS: number[] = [THREE.RGFormat, THREE.RED_GREEN_RGTC2_Format, THREE.RG11_EAC_Format];

/**
 * Normal map without precomputed tangents: the tangent frame comes from screen-space derivatives of the view
 * position and `vgUv` (same construction as three's derivative tangents, http://www.thetenthplanet.de/archives/1180).
 */
export function vgNormalMap(map: THREE.Texture, scale: Node, type: THREE.NormalMapTypes = THREE.TangentSpaceNormalMap): Node {
  const sample: Node = vgTexture(map);
  let n: Node = sample.xyz.mul(2).sub(1);
  if (TWO_CHANNEL_FORMATS.includes(map.format)) {
    const xy = sample.xy.mul(2).sub(1);
    n = vec3(xy, sqrt(float(1).sub(xy.dot(xy)).max(0)));
  }
  n = vec3(n.xy.mul(scale), n.z);
  // Object space is the VirtualMesh's space (world space): correct for instances without rotation.
  if (type === THREE.ObjectSpaceNormalMap) return transformNormalToView(n);

  const q0: Node = (positionView as Node).dFdx();
  const q1: Node = (positionView as Node).dFdy();
  const st0: Node = (vgUv as Node).dFdx();
  const st1: Node = (vgUv as Node).dFdy();
  const N: Node = normalView;
  const q1perp = q1.cross(N);
  const q0perp = N.cross(q0);
  const T = q1perp.mul(st0.x).add(q0perp.mul(st1.x));
  const B = q1perp.mul(st0.y).add(q0perp.mul(st1.y));
  const det = T.dot(T).max(B.dot(B));
  const s = det.equal(0).select(0, det.inverseSqrt());
  return (mat3(negateOnBackSide(T.mul(s)), negateOnBackSide(B.mul(s)), N) as Node).mul(n).normalize();
}

/**
 * Makes a material's textures sample `vgUv`. Called by VirtualMesh on the material it takes over.
 *
 * - `map` and `alphaMap` move into `colorNode` (and are cleared on the material): the shadow pass reads
 *   `colorNode.a` for alpha-tested shadows, but would sample the material's `map` with the missing `uv()`.
 * - Normal, bump and clearcoat normal maps become explicit normal nodes, because three builds normals in a
 *   context that ignores custom UVs.
 * - Every other texture lookup without explicit coordinates (roughness, metalness, emissive, AO, light,
 *   clearcoat, sheen, transmission maps, textures in your own nodes, ...) gets `vgUv` through
 *   `material.contextNode`. A `getUV` in your own `contextNode` takes precedence.
 *
 * Nodes the material already has are kept: only properties three would otherwise read implicitly are rewired.
 */
export function bindVirtualGeometryTextures(material: THREE.NodeMaterial) {
  const m = material as Node;
  if (!m.colorNode && (isTexture(m.map) || isTexture(m.alphaMap))) {
    // With `map` cleared below, materialColor is just the color uniform (live, like three's).
    let color: Node = vec4(materialColor as Node, 1);
    if (isTexture(m.map)) {
      const sample = vgTexture(m.map);
      color = vec4((materialColor as Node).mul(sample.rgb), sample.a);
    }
    if (isTexture(m.alphaMap)) color = vec4(color.rgb, color.a.mul(vgTexture(m.alphaMap).g));
    m.colorNode = color;
    m.map = null;
    m.alphaMap = null;
  }
  if (!m.normalNode) {
    if (isTexture(m.normalMap)) m.normalNode = vgNormalMap(m.normalMap, materialReference('normalScale', 'vec2'), m.normalMapType);
    else if (isTexture(m.bumpMap)) m.normalNode = bumpMap(vgTexture(m.bumpMap).r, materialReference('bumpScale', 'float'));
  }
  if (!m.clearcoatNormalNode && isTexture(m.clearcoatNormalMap)) {
    m.clearcoatNormalNode = vgNormalMap(m.clearcoatNormalMap, materialReference('clearcoatNormalScale', 'vec2'));
  }
  m.contextNode = m.contextNode ? context(m.contextNode, { getUV }) : context({ getUV });
  material.needsUpdate = true;
}

const isTexture = (t: unknown): t is THREE.Texture => (t as THREE.Texture | null)?.isTexture === true;
