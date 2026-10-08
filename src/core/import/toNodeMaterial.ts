import * as THREE from 'three/webgpu';

type MaterialClass = new () => THREE.Material;

/** Classic mesh material type -> [classic class (whose copy() knows every property), node material class]. */
const CONVERSIONS: Record<string, [MaterialClass, new () => THREE.NodeMaterial]> = {
  MeshPhysicalMaterial: [THREE.MeshPhysicalMaterial, THREE.MeshPhysicalNodeMaterial],
  MeshStandardMaterial: [THREE.MeshStandardMaterial, THREE.MeshStandardNodeMaterial],
  MeshPhongMaterial: [THREE.MeshPhongMaterial, THREE.MeshPhongNodeMaterial],
  MeshLambertMaterial: [THREE.MeshLambertMaterial, THREE.MeshLambertNodeMaterial],
  MeshBasicMaterial: [THREE.MeshBasicMaterial, THREE.MeshBasicNodeMaterial],
  MeshToonMaterial: [THREE.MeshToonMaterial, THREE.MeshToonNodeMaterial],
  MeshMatcapMaterial: [THREE.MeshMatcapMaterial, THREE.MeshMatcapNodeMaterial],
  MeshNormalMaterial: [THREE.MeshNormalMaterial, THREE.MeshNormalNodeMaterial],
};

/** Subclasses keep their parent's flags: check the most derived first. */
const FLAGS: [string, string][] = [
  ['isMeshPhysicalMaterial', 'MeshPhysicalMaterial'],
  ['isMeshStandardMaterial', 'MeshStandardMaterial'],
  ['isMeshPhongMaterial', 'MeshPhongMaterial'],
  ['isMeshLambertMaterial', 'MeshLambertMaterial'],
  ['isMeshBasicMaterial', 'MeshBasicMaterial'],
  ['isMeshToonMaterial', 'MeshToonMaterial'],
  ['isMeshMatcapMaterial', 'MeshMatcapMaterial'],
  ['isMeshNormalMaterial', 'MeshNormalMaterial'],
];

function conversionOf(material: THREE.Material) {
  const flags = material as unknown as Record<string, unknown>;
  const type = CONVERSIONS[material.type] ? material.type : FLAGS.find(([flag]) => flags[flag] === true)?.[1];
  return type ? CONVERSIONS[type] : null;
}

/** True when `toNodeMaterial` can convert the material (node materials, and the classic mesh materials). */
export function canConvertToNodeMaterial(material: THREE.Material): boolean {
  return (material as THREE.NodeMaterial).isNodeMaterial === true || conversionOf(material) !== null;
}

/**
 * The node-material equivalent of a classic mesh material: MeshStandardMaterial -> MeshStandardNodeMaterial,
 * and likewise for Physical, Phong, Lambert, Basic, Toon, Matcap and Normal. Every property is copied (colors,
 * maps and their texture objects, transparency, alphaTest, side, blending, userData, ...). Node materials are
 * returned as they are; materials without a node equivalent (ShaderMaterial, ...) return null.
 */
export function toNodeMaterial(material: THREE.Material): THREE.NodeMaterial | null {
  if ((material as THREE.NodeMaterial).isNodeMaterial) return material as THREE.NodeMaterial;
  const conversion = conversionOf(material);
  if (!conversion) return null;
  const [Classic, Node] = conversion;
  const result = new Node();
  const defines = (result as unknown as { defines?: unknown }).defines;
  // The classic copy() lists every property of its material type; the node material has the same ones.
  (Classic.prototype.copy as (this: THREE.Material, source: THREE.Material) => THREE.Material).call(result, material);
  (result as unknown as { defines?: unknown }).defines = defines;
  return result;
}
