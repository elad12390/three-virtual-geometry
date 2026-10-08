/**
 * Binary container for `VirtualMeshData` (`.vgeo`, "virtual geometry").
 *
 * Layout (little-endian):
 *   0   magic 'VGEO'
 *   4   u32 format version
 *   8   u32 header byte length (UTF-8 JSON, then space-padded to 4 bytes)
 *   12  u32 checksum of everything after this 16-byte prefix
 *   16  JSON header
 *   ..  payload: one 4-byte aligned blob per typed-array field
 *
 * The header lists every typed-array field (name, array type, length, encoding, payload offset and size)
 * and carries every other field as JSON, so fields added to `VirtualMeshData` later are stored without
 * format changes. Compression is lossless: meshoptimizer's vertex and index codecs, optionally followed by
 * deflate, whichever is smallest per field. `indices` is not stored: it is rebuilt from the meshlet arrays.
 */
import { MeshoptDecoder, MeshoptEncoder } from 'meshoptimizer';
import { MESHLET_INFO_STRIDE } from '../constants.js';
import type { VirtualMeshData } from '../preprocess/buildVirtualMesh.js';

/** Bumped whenever the container layout or an encoding changes; older readers reject newer files. */
export const VG_FORMAT_VERSION = 1;

const MAGIC = 0x4f454756; // 'VGEO' read as a little-endian u32
const PREFIX_BYTES = 16;

export interface VirtualMeshEncodeOptions {
  /** Store `indices` instead of rebuilding it from the meshlet arrays on decode. Default false. */
  keepIndices?: boolean;
  /** Lossless compression with meshoptimizer's codecs. Default true; false stores every array uncompressed. */
  compress?: boolean;
  /**
   * Also deflate fields where that makes them smaller (about 30% smaller files, about 2x slower decoding).
   * Default true; ignored where `CompressionStream` is unavailable. The IndexedDB cache turns it off.
   */
  deflate?: boolean;
}

type Encoding = 'raw' | 'meshopt-vertex' | 'meshlet-triangles' | 'meshlet-indices';

interface FieldEntry {
  name: string;
  type: TypedArrayName;
  length: number;
  encoding: Encoding;
  /** Bytes per element group for 'meshopt-vertex'. */
  stride?: number;
  /** The encoded blob is additionally deflate-compressed. */
  deflate?: boolean;
  /** Payload-relative byte offset and size of the blob. */
  offset: number;
  byteLength: number;
}

interface Header {
  generator: string;
  /** Every top-level key of the original object, in order. */
  keys: string[];
  fields: FieldEntry[];
  values: Record<string, unknown>;
}

type AnyTypedArray =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

const TYPED_ARRAYS = {
  Int8Array,
  Uint8Array,
  Uint8ClampedArray,
  Int16Array,
  Uint16Array,
  Int32Array,
  Uint32Array,
  Float32Array,
  Float64Array,
};
type TypedArrayName = keyof typeof TYPED_ARRAYS;

/**
 * Per-field encoder hints, from measurements on the demo assets. `strides`: element groups (bytes) to try with
 * the vertex codec (unknown fields use their element size, at least 4). `deflateRaw`: also try deflate on the
 * uncompressed array (meshletBounds repeats whole spheres, which deflate finds and the vertex codec does not).
 */
const FIELD_HINTS: Record<string, { strides?: number[]; deflateRaw?: boolean }> = {
  positions: { strides: [12] },
  normals: { strides: [12] },
  colors: { strides: [12] },
  uvs: { strides: [8] },
  tangents: { strides: [16] },
  meshletBounds: { strides: [64], deflateRaw: true },
  meshletInfo: { strides: [16] },
};

export class VirtualMeshFormatError extends Error {
  constructor(message: string) {
    super(`three-virtual-geometry: ${message}`);
    this.name = 'VirtualMeshFormatError';
  }
}

export async function encodeVirtualMesh(data: VirtualMeshData, options: VirtualMeshEncodeOptions = {}): Promise<Uint8Array> {
  const compress = options.compress ?? true;
  if (compress) await MeshoptEncoder.ready;
  const record = data as unknown as Record<string, unknown>;
  const keys = Object.keys(record).filter((k) => record[k] !== undefined && typeof record[k] !== 'function');
  const fields: FieldEntry[] = [];
  const blobs: Uint8Array[] = [];
  const values: Record<string, unknown> = {};
  let offset = 0;

  for (const name of keys) {
    const value = record[name];
    if (!ArrayBuffer.isView(value)) {
      values[name] = value;
      continue;
    }
    const type = value.constructor.name as TypedArrayName;
    if (!(type in TYPED_ARRAYS)) throw new VirtualMeshFormatError(`cannot serialize field "${name}" of type ${type}`);
    const array = value as AnyTypedArray;
    let blob: Uint8Array;
    let entry: Omit<FieldEntry, 'offset' | 'byteLength'>;
    if (name === 'indices' && !options.keepIndices && indicesMatchMeshlets(data)) {
      entry = { name, type, length: array.length, encoding: 'meshlet-indices' };
      blob = new Uint8Array(0);
    } else if (compress) {
      ({ entry, blob } = await encodeField(name, type, array, data, options.deflate ?? true));
    } else {
      entry = { name, type, length: array.length, encoding: 'raw' };
      blob = bytesOf(array);
    }
    fields.push({ ...entry, offset, byteLength: blob.length });
    blobs.push(blob);
    offset += align4(blob.length);
  }

  const headerJson: Header = { generator: 'three-virtual-geometry', keys, fields, values };
  const headerBytes = new TextEncoder().encode(JSON.stringify(headerJson, jsonReplacer));
  const headerPadded = align4(headerBytes.length);
  const out = new Uint8Array(PREFIX_BYTES + headerPadded + offset);
  out.set(headerBytes, PREFIX_BYTES);
  out.fill(0x20, PREFIX_BYTES + headerBytes.length, PREFIX_BYTES + headerPadded);
  const payloadStart = PREFIX_BYTES + headerPadded;
  fields.forEach((f, i) => out.set(blobs[i], payloadStart + f.offset));
  const view = new DataView(out.buffer);
  view.setUint32(0, MAGIC, true);
  view.setUint32(4, VG_FORMAT_VERSION, true);
  view.setUint32(8, headerBytes.length, true);
  view.setUint32(12, checksum(out, PREFIX_BYTES), true);
  return out;
}

export async function decodeVirtualMesh(input: ArrayBuffer | ArrayBufferView): Promise<VirtualMeshData> {
  let bytes = ArrayBuffer.isView(input) ? new Uint8Array(input.buffer, input.byteOffset, input.byteLength) : new Uint8Array(input);
  if (bytes.byteOffset % 4 !== 0) bytes = bytes.slice(); // checksum and payload views need 4-byte alignment
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length < PREFIX_BYTES || view.getUint32(0, true) !== MAGIC) {
    throw new VirtualMeshFormatError('not a three-virtual-geometry geometry file (bad magic bytes)');
  }
  const version = view.getUint32(4, true);
  if (version !== VG_FORMAT_VERSION) {
    throw new VirtualMeshFormatError(`unsupported geometry format version ${version} (this build reads version ${VG_FORMAT_VERSION}); re-bake the file`);
  }
  const headerLength = view.getUint32(8, true);
  const payloadStart = PREFIX_BYTES + align4(headerLength);
  if (payloadStart > bytes.length || bytes.length % 4 !== 0) throw new VirtualMeshFormatError('geometry file is truncated');
  // Header and field bounds are checked before the checksum so truncation gets its own message.

  let header: Header;
  try {
    header = JSON.parse(new TextDecoder().decode(bytes.subarray(PREFIX_BYTES, PREFIX_BYTES + headerLength)), jsonReviver);
  } catch {
    throw new VirtualMeshFormatError('geometry file is corrupt or truncated (unreadable header)');
  }
  if (!header || !Array.isArray(header.fields) || !Array.isArray(header.keys) || typeof header.values !== 'object') {
    throw new VirtualMeshFormatError('geometry file is corrupt (malformed header)');
  }

  const payload = bytes.subarray(payloadStart);
  const arrays: Record<string, AnyTypedArray> = {};
  // Derived encodings read other fields, so they go last.
  const order = [...header.fields].sort((a, b) => rank(a.encoding) - rank(b.encoding));
  if (order.some((f) => f.encoding !== 'raw')) await MeshoptDecoder.ready;
  for (const field of order) {
    const Ctor = TYPED_ARRAYS[field.type];
    if (!Ctor || !Number.isInteger(field.length) || field.length < 0) throw new VirtualMeshFormatError(`geometry file is corrupt (field "${field.name}")`);
    if (field.offset % 4 !== 0 || field.offset + field.byteLength > payload.length) {
      throw new VirtualMeshFormatError(`geometry file is truncated (field "${field.name}")`);
    }
  }
  if (checksum(bytes, PREFIX_BYTES) !== view.getUint32(12, true)) throw new VirtualMeshFormatError('geometry file is corrupt (checksum mismatch)');
  const fail = (field: FieldEntry, e: unknown) =>
    e instanceof VirtualMeshFormatError ? e : new VirtualMeshFormatError(`geometry file is corrupt (field "${field.name}": ${(e as Error).message})`);
  // Inflate every deflated field at once (runs concurrently where the platform allows), then decode in order.
  const blobs = await Promise.all(
    order.map((f) => {
      const blob = payload.subarray(f.offset, f.offset + f.byteLength);
      return f.deflate ? inflate(blob).catch((e) => Promise.reject(fail(f, e))) : blob;
    })
  );
  for (let i = 0; i < order.length; i++) {
    try {
      arrays[order[i].name] = await decodeField({ ...order[i], deflate: false }, blobs[i], arrays);
    } catch (e) {
      throw fail(order[i], e);
    }
  }

  const result: Record<string, unknown> = {};
  for (const key of header.keys) result[key] = key in arrays ? arrays[key] : header.values[key];
  return result as unknown as VirtualMeshData;
}

/** Fetches and decodes a baked `.vgeo` file, or decodes bytes that are already in memory. */
export async function loadVirtualMesh(source: string | URL | ArrayBuffer | ArrayBufferView, init?: RequestInit): Promise<VirtualMeshData> {
  if (typeof source !== 'string' && !(source instanceof URL)) return decodeVirtualMesh(source);
  const response = await fetch(source, init);
  if (!response.ok) throw new Error(`three-virtual-geometry: failed to load ${source} (HTTP ${response.status})`);
  return decodeVirtualMesh(await response.arrayBuffer());
}

// ------------------------------------------------------------------ encoding

async function encodeField(name: string, type: TypedArrayName, array: AnyTypedArray, data: VirtualMeshData, useDeflate: boolean) {
  const raw = bytesOf(array);
  const base = { name, type, length: array.length };
  const hint = FIELD_HINTS[name] ?? {};
  const rawCandidate = { entry: { ...base, encoding: 'raw' as Encoding }, blob: raw };
  const candidates: { entry: Omit<FieldEntry, 'offset' | 'byteLength'>; blob: Uint8Array }[] = [rawCandidate];

  // Always much smaller than the vertex codec on meshlet triangles, so that one is only a fallback there.
  const triangles = name === 'meshletTriangles' && array instanceof Uint32Array ? encodeMeshletTriangles(array, data.meshletInfo) : null;
  if (triangles) candidates.push({ entry: { ...base, encoding: 'meshlet-triangles' }, blob: triangles });
  else if (raw.length % 4 === 0 && raw.length > 0) {
    const strides = (hint.strides ?? [Math.max(4, array.BYTES_PER_ELEMENT)]).filter((s) => s % 4 === 0 && s <= 256 && raw.length % s === 0);
    for (const stride of strides) {
      const blob = MeshoptEncoder.encodeVertexBufferLevel(raw, raw.length / stride, stride, 2, 1);
      candidates.push({ entry: { ...base, encoding: 'meshopt-vertex', stride }, blob });
    }
  }
  candidates.sort((a, b) => a.blob.length - b.blob.length);
  if (useDeflate && canDeflate() && raw.length >= 64) {
    // Deflate is slow to run on everything: try it on the best codec output (and on the raw data if hinted).
    const toDeflate = hint.deflateRaw && candidates[0] !== rawCandidate ? [candidates[0], rawCandidate] : [candidates[0]];
    for (const c of toDeflate) candidates.push({ entry: { ...c.entry, deflate: true }, blob: await deflate(c.blob) });
    candidates.sort((a, b) => a.blob.length - b.blob.length);
  }

  for (const c of candidates) {
    // The chosen encoding is decoded once and compared, so a stored file always round-trips exactly.
    const decoded = await decodeField({ ...c.entry, offset: 0, byteLength: c.blob.length }, c.blob, { meshletInfo: data.meshletInfo });
    if (sameBytes(bytesOf(decoded), raw)) return c;
  }
  return rawCandidate;
}

/**
 * Meshlet triangles as one index buffer over meshlet-vertex slots (vertexOffset + local id). Meshlet-local
 * ids are in first-use order, so that buffer is ideal input for the index codec. The codec may rotate the
 * corners of a triangle, so each triangle's rotation is stored too (2 bits) to keep the exact corner order.
 * Layout: u32 codec byte length, codec bytes (padded to 4), packed rotations.
 */
function encodeMeshletTriangles(triangles: Uint32Array, meshletInfo: Uint32Array | undefined): Uint8Array | null {
  if (!(meshletInfo instanceof Uint32Array)) return null;
  const slots = new Uint32Array(triangles.length * 3);
  const covered = new Uint8Array(triangles.length);
  for (let m = 0; m < meshletInfo.length; m += MESHLET_INFO_STRIDE) {
    const count = meshletInfo[m];
    const first = meshletInfo[m + 1] / 3;
    const vertexOffset = meshletInfo[m + 3];
    if (first + count > triangles.length) return null;
    for (let t = first; t < first + count; t++) {
      const packed = triangles[t];
      if (covered[t] || packed >>> 24 !== 0) return null;
      covered[t] = 1;
      slots[t * 3] = vertexOffset + (packed & 255);
      slots[t * 3 + 1] = vertexOffset + ((packed >> 8) & 255);
      slots[t * 3 + 2] = vertexOffset + ((packed >> 16) & 255);
    }
  }
  if (!covered.every((c) => c === 1)) return null;

  const codec = MeshoptEncoder.encodeIndexBuffer(bytesOf(slots), slots.length, 4);
  const decoded = new Uint32Array(slots.length);
  MeshoptDecoder.decodeIndexBuffer(bytesOf(decoded), slots.length, 4, codec);
  const rotations = new Uint8Array(Math.ceil(triangles.length / 4));
  for (let t = 0; t < triangles.length; t++) {
    const a = slots[t * 3], b = slots[t * 3 + 1], c = slots[t * 3 + 2];
    const x = decoded[t * 3], y = decoded[t * 3 + 1], z = decoded[t * 3 + 2];
    let r: number;
    if (a === x && b === y && c === z) r = 0;
    else if (a === y && b === z && c === x) r = 1;
    else if (a === z && b === x && c === y) r = 2;
    else return null;
    rotations[t >> 2] |= r << ((t & 3) * 2);
  }
  const out = new Uint8Array(4 + align4(codec.length) + rotations.length);
  new DataView(out.buffer).setUint32(0, codec.length, true);
  out.set(codec, 4);
  out.set(rotations, 4 + align4(codec.length));
  return out;
}

/** True if `indices` is exactly what decoding rebuilds from meshletInfo / meshletVertices / meshletTriangles. */
function indicesMatchMeshlets(data: VirtualMeshData): boolean {
  const { indices, meshletInfo, meshletVertices, meshletTriangles } = data;
  if (!(indices instanceof Uint32Array && meshletInfo instanceof Uint32Array && meshletVertices instanceof Uint32Array && meshletTriangles instanceof Uint32Array)) {
    return false;
  }
  try {
    return sameBytes(bytesOf(rebuildIndices(indices.length, meshletInfo, meshletVertices, meshletTriangles)), bytesOf(indices));
  } catch {
    return false;
  }
}

// ------------------------------------------------------------------ decoding

const rank = (e: Encoding) => (e === 'meshlet-triangles' ? 1 : e === 'meshlet-indices' ? 2 : 0);

async function decodeField(field: FieldEntry, blob: Uint8Array, arrays: Record<string, AnyTypedArray>): Promise<AnyTypedArray> {
  const Ctor = TYPED_ARRAYS[field.type];
  const out = new Ctor(field.length);
  const target = bytesOf(out);
  const src = field.deflate ? await inflate(blob) : blob;
  switch (field.encoding) {
    case 'raw':
      if (src.length !== target.length) throw new VirtualMeshFormatError(`geometry file is corrupt (field "${field.name}" has the wrong size)`);
      target.set(src);
      break;
    case 'meshopt-vertex': {
      const stride = field.stride ?? 0;
      if (stride <= 0 || stride % 4 !== 0 || target.length % stride !== 0) throw new VirtualMeshFormatError(`geometry file is corrupt (field "${field.name}" stride)`);
      MeshoptDecoder.decodeVertexBuffer(target, target.length / stride, stride, src);
      break;
    }
    case 'meshlet-triangles':
      decodeMeshletTriangles(out as Uint32Array, src, requireArray(arrays, 'meshletInfo', field.name));
      break;
    case 'meshlet-indices':
      out.set(
        rebuildIndices(
          field.length,
          requireArray(arrays, 'meshletInfo', field.name),
          requireArray(arrays, 'meshletVertices', field.name),
          requireArray(arrays, 'meshletTriangles', field.name)
        )
      );
      break;
    default:
      throw new VirtualMeshFormatError(`unknown encoding "${field.encoding}" for field "${field.name}"`);
  }
  return out;
}

function requireArray(arrays: Record<string, AnyTypedArray>, name: string, forField: string): Uint32Array {
  const a = arrays[name];
  if (!(a instanceof Uint32Array)) throw new VirtualMeshFormatError(`geometry file is corrupt ("${forField}" needs "${name}")`);
  return a;
}

function decodeMeshletTriangles(out: Uint32Array, blob: Uint8Array, meshletInfo: Uint32Array) {
  const codecLength = new DataView(blob.buffer, blob.byteOffset, blob.byteLength).getUint32(0, true);
  const rotationsStart = 4 + align4(codecLength);
  if (rotationsStart + Math.ceil(out.length / 4) > blob.length) throw new Error('truncated triangle data');
  const slots = new Uint32Array(out.length * 3);
  MeshoptDecoder.decodeIndexBuffer(bytesOf(slots), slots.length, 4, blob.subarray(4, 4 + codecLength));
  const rotations = blob.subarray(rotationsStart);
  for (let m = 0; m < meshletInfo.length; m += MESHLET_INFO_STRIDE) {
    const count = meshletInfo[m];
    const first = meshletInfo[m + 1] / 3;
    const vertexOffset = meshletInfo[m + 3];
    if (first + count > out.length) throw new Error('meshlet range out of bounds');
    for (let t = first; t < first + count; t++) {
      const r = (rotations[t >> 2] >> ((t & 3) * 2)) & 3;
      const x = slots[t * 3] - vertexOffset, y = slots[t * 3 + 1] - vertexOffset, z = slots[t * 3 + 2] - vertexOffset;
      const [a, b, c] = r === 0 ? [x, y, z] : r === 1 ? [y, z, x] : [z, x, y];
      if ((a | b | c) >>> 8 !== 0) throw new Error('meshlet-local vertex id out of range');
      out[t] = a | (b << 8) | (c << 16);
    }
  }
}

/** `indices[firstIndex + 3t + k]` = global vertex of corner k of the meshlet's triangle t. */
function rebuildIndices(length: number, meshletInfo: Uint32Array, meshletVertices: Uint32Array, meshletTriangles: Uint32Array): Uint32Array {
  const indices = new Uint32Array(length);
  for (let m = 0; m < meshletInfo.length; m += MESHLET_INFO_STRIDE) {
    const count = meshletInfo[m];
    const first = meshletInfo[m + 1];
    const vertexOffset = meshletInfo[m + 3];
    if (first + count * 3 > length || first / 3 + count > meshletTriangles.length) throw new VirtualMeshFormatError('geometry file is corrupt (meshlet range out of bounds)');
    for (let t = 0; t < count; t++) {
      const packed = meshletTriangles[first / 3 + t];
      for (let k = 0; k < 3; k++) {
        const v = vertexOffset + ((packed >> (8 * k)) & 255);
        if (v >= meshletVertices.length) throw new VirtualMeshFormatError('geometry file is corrupt (meshlet vertex out of bounds)');
        indices[first + t * 3 + k] = meshletVertices[v];
      }
    }
  }
  return indices;
}

// ------------------------------------------------------------------ helpers

const align4 = (n: number) => (n + 3) & ~3;

function bytesOf(a: ArrayBufferView): Uint8Array {
  return new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
}

function sameBytes(a: Uint8Array, b: Uint8Array) {
  if (a.length !== b.length) return false;
  if (a.byteOffset % 4 === 0 && b.byteOffset % 4 === 0 && a.length % 4 === 0) {
    const x = new Uint32Array(a.buffer, a.byteOffset, a.length / 4);
    const y = new Uint32Array(b.buffer, b.byteOffset, b.length / 4);
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
    return true;
  }
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** 32-bit multiplicative hash over the u32 words from `start` (catches truncation and bit rot, not tampering). */
function checksum(bytes: Uint8Array, start: number): number {
  const words = new Uint32Array(bytes.buffer, bytes.byteOffset + start, (bytes.length - start) >> 2);
  let h = 0x811c9dc5 ^ words.length;
  for (let i = 0; i < words.length; i++) {
    h = Math.imul(h ^ words[i], 0x01000193);
    h ^= h >>> 15;
  }
  return h >>> 0;
}

const canDeflate = () => typeof CompressionStream !== 'undefined' && typeof DecompressionStream !== 'undefined';

async function deflate(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new CompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function inflate(bytes: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === 'undefined') throw new VirtualMeshFormatError('this environment cannot decompress the geometry file (no DecompressionStream)');
  const stream = new Blob([bytes as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** JSON has no Infinity/NaN; error fields use them (e.g. root parent errors). */
function jsonReplacer(key: string, value: unknown) {
  if (typeof value === 'number' && !Number.isFinite(value)) return { $number: String(value) };
  if (ArrayBuffer.isView(value)) throw new VirtualMeshFormatError(`typed array nested inside field "${key}" cannot be serialized; store it as a top-level field`);
  return value;
}

function jsonReviver(_key: string, value: unknown) {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const keys = Object.keys(value);
    if (keys.length === 1 && keys[0] === '$number') return Number((value as { $number: string }).$number);
  }
  return value;
}
