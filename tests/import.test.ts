import { expect, test } from 'vitest';
import * as THREE from 'three/webgpu';
import { buildVirtualMesh, collectVirtualGeometryGroups, fromBufferGeometry, VirtualGeometry, toNodeMaterial } from '../src/index';
import { buildVoxelProxy } from '../src/core/preprocess/voxelProxy';
import { packUvs } from '../src/core/runtime/VirtualMesh';

test('glTF-style scene import: grouping, materials, UV seams', async () => {
  const assert = (cond: unknown, msg: string) => {
    expect(cond, msg).toBeTruthy();
  };
  const near = (a: number, b: number, eps = 1e-5) => Math.abs(a - b) <= eps;

  // ---------- UV welding ----------
  {
    const sphere = new THREE.SphereGeometry(1, 32, 16);
    const src = fromBufferGeometry(sphere);
    const byPosition = fromBufferGeometry(sphere, { uvs: false });
    assert(src.uvs && src.uvs.length === (src.positions.length / 3) * 2, 'uv weld: one uv per welded vertex');
    assert(src.positions.length > byPosition.positions.length, 'uv weld: seam vertices stay separate (more vertices than a position weld)');
    assert(src.indices.length === sphere.index!.count, 'uv weld: every corner kept');

    // Every corner keeps exactly its own position and uv: nothing was merged across different UVs.
    const p = sphere.attributes.position;
    const uv = sphere.attributes.uv;
    let exact = true;
    for (let i = 0; i < sphere.index!.count; i++) {
      const v = sphere.index!.getX(i);
      const w = src.indices[i];
      if (!near(src.uvs![w * 2], uv.getX(v)) || !near(src.uvs![w * 2 + 1], uv.getY(v))) exact = false;
      if (src.positions[w * 3] !== p.getX(v) || src.positions[w * 3 + 1] !== p.getY(v) || src.positions[w * 3 + 2] !== p.getZ(v)) exact = false;
    }
    assert(exact, 'uv weld: each corner keeps its own position and uv');

    // Smooth normals: copies of a vertex on a seam share the normal of their position.
    const normalOf = new Map<string, number[]>();
    let smooth = true;
    for (let w = 0; w < src.positions.length / 3; w++) {
      const key = Array.from(src.positions.subarray(w * 3, w * 3 + 3)).join(',');
      const n = Array.from(src.normals.subarray(w * 3, w * 3 + 3));
      const seen = normalOf.get(key);
      if (seen && !seen.every((x, k) => x === n[k])) smooth = false;
      normalOf.set(key, n);
    }
    assert(smooth, 'uv weld: seam copies share one smooth normal');

    const box = new THREE.BoxGeometry(1, 1, 1);
    const kept = fromBufferGeometry(box, { normals: 'keep' });
    assert(kept.positions.length / 3 === 24, "normals 'keep': a box keeps its 24 hard-edged vertices");
    const half = fromBufferGeometry(box, { uvs: false, range: { start: 6, count: 12 } }); // two adjacent faces
    assert(half.indices.length === 12 && half.positions.length / 3 === 6, 'range: only the given index range is welded');

    const g = new THREE.TorusGeometry(2, 0.6, 48, 128);
    const torus = fromBufferGeometry(g);
    const data = await buildVirtualMesh(torus);
    assert(data.stats.leafTriangles === g.index!.count / 3, 'uv build: leaf triangle count preserved');
    assert(data.uvs !== null && data.uvs.length === (data.positions.length / 3) * 2, 'uv build: uvs cover every vertex (incl. voxel ones)');
    assert(data.stats.rootTriangles < data.stats.leafTriangles / 50, `uv build: simplifies across seams (roots ${data.stats.rootTriangles} tris)`);

    const uvSphere = await buildVirtualMesh(fromBufferGeometry(new THREE.SphereGeometry(1, 128, 64)));
    assert(uvSphere.stats.rootTriangles <= 64, `uv build: a UV sphere (pole seam fans) still reduces to a few triangles (${uvSphere.stats.rootTriangles})`);
  }

  // ---------- voxel proxy uvs ----------
  {
    const g = new THREE.SphereGeometry(1, 24, 12);
    const s = fromBufferGeometry(g);
    const proxy = buildVoxelProxy(s.positions, null, s.indices, 16, s.uvs)!;
    const finite = proxy.uvs!.every((x) => Number.isFinite(x) && x >= -1e-6 && x <= 1 + 1e-6);
    assert(proxy.uvs!.length === (proxy.positions.length / 3) * 2 && finite, 'voxel proxy: every vertex gets a source uv');
  }

  // ---------- uv packing ----------
  {
    const uvs = new Float32Array([0, 0, 1, 1, 0.123456, 0.987654, -2, 3, 5, -1]);
    const { packed, scale, offset } = packUvs(uvs);
    let maxError = 0;
    for (let i = 0; i < packed.length; i++) {
      const u = ((packed[i] & 0xffff) / 65535) * scale[0] + offset[0];
      const v = ((packed[i] >>> 16) / 65535) * scale[1] + offset[1];
      maxError = Math.max(maxError, Math.abs(u - uvs[i * 2]), Math.abs(v - uvs[i * 2 + 1]));
    }
    assert(maxError <= (Math.max(...scale) / 65535) * 0.5 + 1e-7, `packUvs: round trip within half a step (${maxError.toExponential(2)})`);
  }

  // ---------- toNodeMaterial ----------
  {
    const map = new THREE.Texture();
    const normalMap = new THREE.Texture();
    const standard = new THREE.MeshStandardMaterial({
      name: 'paint',
      color: 0x336699,
      roughness: 0.3,
      metalness: 0.7,
      map,
      normalMap,
      normalScale: new THREE.Vector2(0.5, -0.5),
      emissive: 0x110000,
      emissiveIntensity: 2,
      transparent: true,
      opacity: 0.6,
      alphaTest: 0.25,
      side: THREE.DoubleSide,
      vertexColors: true,
      flatShading: true,
    });
    standard.userData = { tag: 'x' };
    const n = toNodeMaterial(standard) as THREE.MeshStandardNodeMaterial;
    assert(n instanceof THREE.MeshStandardNodeMaterial, 'toNodeMaterial: MeshStandardMaterial -> MeshStandardNodeMaterial');
    assert(n.color.getHex() === 0x336699 && n.color !== standard.color, 'toNodeMaterial: color copied (not shared)');
    assert(n.roughness === 0.3 && n.metalness === 0.7, 'toNodeMaterial: roughness/metalness');
    assert(n.map === map && n.normalMap === normalMap, 'toNodeMaterial: same texture objects');
    assert(n.normalScale.x === 0.5 && n.normalScale.y === -0.5, 'toNodeMaterial: normalScale');
    assert(n.emissive.getHex() === 0x110000 && n.emissiveIntensity === 2, 'toNodeMaterial: emissive');
    assert(n.transparent && n.opacity === 0.6 && n.alphaTest === 0.25 && n.side === THREE.DoubleSide, 'toNodeMaterial: transparency, alphaTest, side');
    assert(n.vertexColors && n.flatShading && n.name === 'paint' && n.userData.tag === 'x', 'toNodeMaterial: vertexColors, flatShading, name, userData');
    assert(n.uuid !== standard.uuid && n.type === 'MeshStandardNodeMaterial', 'toNodeMaterial: a new material of the node type');

    const physical = new THREE.MeshPhysicalMaterial({ clearcoat: 0.8, transmission: 0.5, sheen: 1, sheenColor: 0xff0000, ior: 1.3 });
    const p = toNodeMaterial(physical) as THREE.MeshPhysicalNodeMaterial;
    assert(p instanceof THREE.MeshPhysicalNodeMaterial && p.clearcoat === 0.8 && p.transmission === 0.5 && p.sheenColor.getHex() === 0xff0000 && p.ior === 1.3, 'toNodeMaterial: physical properties');
    const phong = toNodeMaterial(new THREE.MeshPhongMaterial({ shininess: 77, specular: 0x123456 })) as THREE.MeshPhongNodeMaterial;
    assert(phong instanceof THREE.MeshPhongNodeMaterial && phong.shininess === 77 && phong.specular.getHex() === 0x123456, 'toNodeMaterial: phong');
    assert(toNodeMaterial(new THREE.MeshLambertMaterial()) instanceof THREE.MeshLambertNodeMaterial, 'toNodeMaterial: lambert');
    const basic = toNodeMaterial(new THREE.MeshBasicMaterial({ color: 0x00ff00, map })) as THREE.MeshBasicNodeMaterial;
    assert(basic instanceof THREE.MeshBasicNodeMaterial && basic.map === map && basic.color.getHex() === 0x00ff00, 'toNodeMaterial: basic');
    const node = new THREE.MeshStandardNodeMaterial();
    assert(toNodeMaterial(node) === node, 'toNodeMaterial: node materials returned as they are');
    assert(toNodeMaterial(new THREE.ShaderMaterial()) === null, 'toNodeMaterial: ShaderMaterial has no equivalent (null)');
  }

  // ---------- Object3D grouping ----------
  {
    const rock = new THREE.IcosahedronGeometry(1, 3);
    const grey = new THREE.MeshStandardMaterial({ color: 0x888888 });
    const red = new THREE.MeshStandardMaterial({ color: 0xff0000 });
    const root = new THREE.Group();
    root.position.set(10, 0, 0);
    const parent = new THREE.Group();
    parent.rotation.y = Math.PI / 2;
    parent.scale.setScalar(2);
    root.add(parent);
    const a = new THREE.Mesh(rock, grey);
    a.position.set(1, 2, 3);
    const b = new THREE.Mesh(rock, grey);
    b.position.set(-4, 0, 0);
    b.castShadow = true;
    const c = new THREE.Mesh(rock, grey);
    c.castShadow = true;
    parent.add(a, b);
    root.add(c);
    const d = new THREE.Mesh(rock, red); // same geometry, other material
    root.add(d);

    // Multi-material: two geometry groups, two materials.
    const box = new THREE.BoxGeometry(1, 1, 1);
    box.clearGroups();
    box.addGroup(0, 18, 0);
    box.addGroup(18, 18, 1);
    const multi = new THREE.Mesh(box, [grey, red]);
    root.add(multi);

    // InstancedMesh with instance colors.
    const inst = new THREE.InstancedMesh(rock, grey, 5);
    inst.position.set(0, 100, 0);
    const m = new THREE.Matrix4();
    for (let i = 0; i < 5; i++) {
      inst.setMatrixAt(i, m.makeTranslation(i, 0, 0));
      inst.setColorAt(i, new THREE.Color(i / 4, 0, 1));
    }
    root.add(inst);

    // Left alone: skinned, morph targets, points, lines, filtered, shader material.
    const skinned = new THREE.SkinnedMesh(rock, grey);
    const morphGeometry = rock.clone();
    morphGeometry.morphAttributes.position = [rock.attributes.position.clone()];
    const morph = new THREE.Mesh(morphGeometry, grey);
    const points = new THREE.Points(rock, new THREE.PointsMaterial());
    const line = new THREE.Line(rock, new THREE.LineBasicMaterial());
    const filtered = new THREE.Mesh(rock, grey);
    filtered.name = 'keep-me';
    const shader = new THREE.Mesh(rock, new THREE.ShaderMaterial());
    root.add(skinned, morph, points, line, filtered, shader);

    // Hidden subtree.
    const hiddenParent = new THREE.Group();
    hiddenParent.visible = false;
    hiddenParent.add(new THREE.Mesh(rock, grey));
    root.add(hiddenParent);

    const { groups, skipped, meshes } = collectVirtualGeometryGroups(root, (mesh) => mesh.name !== 'keep-me');
    const find = (pred: (g: (typeof groups)[number]) => boolean) => groups.filter(pred);
    const shadowed = find((g) => g.geometry === rock && g.material === grey && g.castShadow && g.visible);
    assert(shadowed.length === 1 && shadowed[0].instances.length === 2, 'grouping: same geometry+material+flags -> one group (b, c)');
    // `a` and the InstancedMesh share geometry, material and flags: one group, 1 + 5 instances.
    const plain = find((g) => g.geometry === rock && g.material === grey && !g.castShadow && g.visible);
    assert(plain.length === 1 && plain[0].instances.length === 6 && plain[0].instances[0].object === a, 'grouping: different castShadow -> separate group; Mesh and InstancedMesh merge');
    assert(find((g) => g.material === red && g.geometry === rock).length === 1, 'grouping: other material -> separate group');

    const worldOk = (g: (typeof groups)[number]) =>
      g.instances.every(({ object, index }, i) => index >= 0 || object.matrixWorld.elements.every((v, k) => near(v, g.matrices[i * 16 + k])));
    assert(groups.every(worldOk), 'grouping: instance matrices are the world matrices');
    const expectedA = new THREE.Matrix4().makeTranslation(10, 0, 0).multiply(new THREE.Matrix4().compose(new THREE.Vector3(), new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.PI / 2, 0)), new THREE.Vector3(2, 2, 2))).multiply(new THREE.Matrix4().makeTranslation(1, 2, 3));
    assert(expectedA.elements.every((v, k) => near(v, plain[0].matrices[k])), 'grouping: nested parent transforms are applied');

    const boxGroups = find((g) => g.geometry === box);
    assert(
      boxGroups.length === 2 && boxGroups.some((g) => g.material === grey && g.range.start === 0 && g.range.count === 18) && boxGroups.some((g) => g.material === red && g.range.start === 18),
      'grouping: multi-material mesh splits into one group per geometry group'
    );

    const instanced = plain[0];
    assert(instanced.instances.slice(1).every(({ object, index }, i) => object === inst && index === i), 'grouping: InstancedMesh -> one instance per instance');
    const i3 = new THREE.Matrix4().makeTranslation(10, 100, 0).multiply(new THREE.Matrix4().makeTranslation(3, 0, 0));
    assert(i3.elements.every((v, k) => near(v, instanced.matrices[(1 + 3) * 16 + k])), 'grouping: InstancedMesh matrices = mesh world * instance matrix');
    const tint = instanced.colors!;
    assert(tint[0] === 1 && tint[1] === 1 && near(tint[(1 + 3) * 4], 0.75) && tint[(1 + 3) * 4 + 3] === 1, 'grouping: instance colors become instance tints (white for plain meshes)');

    const reasons = new Map(skipped.map((s) => [s.object, s.reason]));
    assert(reasons.get(skinned) === 'skinned' && reasons.get(morph) === 'morph targets', 'grouping: skinned and morph meshes left alone');
    assert(reasons.get(filtered) === 'filtered out' && reasons.get(shader) !== undefined, 'grouping: filter and unconvertible materials respected');
    assert(!meshes.includes(points as unknown as THREE.Mesh) && !meshes.includes(line as unknown as THREE.Mesh), 'grouping: points and lines ignored');
    assert(find((g) => !g.visible).length === 1, 'grouping: hidden subtree -> invisible group');
  }

  // ---------- full import (CPU side; VirtualMesh only builds node graphs here) ----------
  {
    const rock = new THREE.IcosahedronGeometry(1, 3);
    const material = new THREE.MeshStandardMaterial({ map: new THREE.Texture() });
    const root = new THREE.Group();
    const meshes = [0, 1, 2].map((i) => {
      const mesh = new THREE.Mesh(rock, material);
      mesh.position.x = i * 3;
      mesh.castShadow = true;
      root.add(mesh);
      return mesh;
    });
    const vg = new VirtualGeometry();
    let builds = 0;
    const progress: number[] = [];
    const result = await vg.add(root, {
      builder: (src, opts) => {
        builds++;
        return buildVirtualMesh(src, opts);
      },
      onProgress: (f) => {
        progress.push(f);
      },
      mesh: { maxDrawDistance: 500 },
    });
    assert(builds === 1 && result.meshes.length === 1 && result.meshes[0].instanceCount === 3, 'import: one build, one VirtualMesh with 3 instances');
    assert(result.meshes[0].castShadow && result.meshes[0].maxDrawDistance === 500, 'import: flags and mesh options applied');
    assert(result.meshes[0].material !== (material as unknown) && material.map !== null, 'import: source material untouched');
    assert(result.meshes[0].material.colorNode !== null, 'import: map rewired into colorNode');
    assert(meshes.every((m) => m.parent === null || m.layers.mask === 0) && result.object.parent === root, 'import: originals out of the graph or hidden, meshes added');
    assert(progress.length > 0 && progress[progress.length - 1] === 1, 'import: progress reaches 1');
    const again = await vg.add(root);
    assert(again.meshes.length === 0, 'import: a second add() does not convert twice');
    result.dispose();
    assert(meshes.every((m) => m.layers.mask === 1) && result.object.parent === null && vg.meshes.length === 0, 'import: dispose restores originals');
  }
}, 300_000);
