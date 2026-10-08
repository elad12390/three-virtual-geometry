/**
 * `buildVirtualMeshCached`: `buildVirtualMesh` behind a persistent IndexedDB cache, keyed by a SHA-256 of the
 * source arrays and build options. The first load builds and stores the encoded result; later loads decode
 * it. Any cache problem (no IndexedDB, private mode, quota, corrupt entry) falls back to building.
 */
import { buildVirtualMesh, type VirtualMeshBuildOptions, type VirtualMeshData, type VirtualMeshSource } from '../preprocess/buildVirtualMesh.js';
import { decodeVirtualMesh, encodeVirtualMesh, VG_FORMAT_VERSION } from './serialize.js';

/**
 * Part of every cache key. Bump it whenever `buildVirtualMesh` produces different output for the same input
 * (algorithm or default changes), so stale cached builds are never returned.
 */
export const VG_BUILD_VERSION = 3;

export interface VirtualGeometryCacheOptions {
  /**
   * Fixed cache key instead of a content hash (skips hashing the source). You must change it whenever the
   * source mesh or build options change.
   */
  key?: string;
  /** 'indexeddb' (default) or false to always build. */
  store?: 'indexeddb' | false;
  /**
   * Least recently used entries are evicted when the cache grows above this many bytes. Default 512 MB. Raise it for
   * scenes with many multi-million-triangle meshes, so every build stays cached; the browser's own storage quota
   * still applies (a write that exceeds it is skipped, never an error).
   */
  maxBytes?: number;
}

export interface VirtualGeometryCachedBuildOptions extends VirtualMeshBuildOptions {
  cache?: VirtualGeometryCacheOptions;
}

const DB_NAME = 'three-virtual-geometry-cache';
const DATA_STORE = 'meshes';
const META_STORE = 'meta';
/** Least recently used entries are evicted above this total size (unless `cache.maxBytes` says otherwise). */
const MAX_CACHE_BYTES = 512 * 1024 * 1024;
const OPEN_TIMEOUT_MS = 3000;

interface MetaEntry {
  size: number;
  lastUsed: number;
}

/**
 * Same as `buildVirtualMesh`, plus a persistent cache. On a hit the build is skipped, `onProgress(1)` is still
 * called, and `stats.buildMs` reports the time this call took (hashing, lookup and decoding).
 */
export async function buildVirtualMeshCached(source: VirtualMeshSource, options: VirtualGeometryCachedBuildOptions = {}): Promise<VirtualMeshData> {
  const { cache, ...buildOptions } = options;
  if (cache?.store === false || !hasIndexedDb()) return buildVirtualMesh(source, buildOptions);

  const t0 = performance.now();
  const key = await virtualMeshCacheKey(source, buildOptions, cache?.key).catch(() => null);
  const db = key ? await openDb() : null;
  if (!db || !key) return buildVirtualMesh(source, buildOptions);

  const bytes = await readEntry(db, key);
  if (bytes) {
    try {
      const data = await decodeVirtualMesh(bytes);
      if (data.stats) data.stats.buildMs = performance.now() - t0;
      touchEntry(db, key);
      await buildOptions.onProgress?.(1);
      return data;
    } catch {
      deleteEntry(db, key); // unreadable (e.g. written by another format version): rebuild below
    }
  }

  const data = await buildVirtualMesh(source, buildOptions);
  try {
    // No deflate: the cache optimizes for decode speed (deflate saves ~30% space but doubles decode time).
    const encoded = await encodeVirtualMesh(data, { deflate: false });
    void writeEntry(db, key, encoded, cache?.maxBytes); // not awaited: the build result is ready, storing happens in the background
  } catch {
    // never fail a build because of the cache
  }
  return data;
}

/** Deletes every cached build. Resolves even if IndexedDB is unavailable. */
export async function clearVirtualMeshCache(): Promise<void> {
  if (!hasIndexedDb()) return;
  const db = await openDb();
  if (!db) return;
  await new Promise<void>((resolve) => {
    try {
      const tx = db.transaction([DATA_STORE, META_STORE], 'readwrite');
      tx.objectStore(DATA_STORE).clear();
      tx.objectStore(META_STORE).clear();
      tx.oncomplete = tx.onerror = tx.onabort = () => resolve();
    } catch {
      resolve();
    }
  });
}

// ------------------------------------------------------------------ key

/** SHA-256 over every source field (typed arrays by content), the build options and the format versions. */
export async function virtualMeshCacheKey(source: VirtualMeshSource, options: VirtualMeshBuildOptions = {}, customKey?: string): Promise<string> {
  const versions = `v${VG_BUILD_VERSION}.${VG_FORMAT_VERSION}`;
  if (customKey !== undefined) return `${versions}:key:${customKey}`;
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error('crypto.subtle unavailable');

  const record = source as unknown as Record<string, unknown>;
  const names = Object.keys(record)
    .filter((k) => record[k] !== undefined && record[k] !== null)
    .sort();
  const describe: unknown[] = [];
  const parts: Promise<ArrayBuffer>[] = [];
  for (const name of names) {
    const value = record[name];
    if (ArrayBuffer.isView(value)) {
      describe.push([name, value.constructor.name, value.byteLength]);
      // Hashed per array, then combined: avoids concatenating hundreds of MB into one buffer.
      parts.push(subtle.digest('SHA-256', value as ArrayBufferView<ArrayBuffer>));
    } else {
      describe.push([name, value]);
    }
  }
  const optionEntries = Object.entries(options)
    .filter(([, v]) => v !== undefined && typeof v !== 'function')
    .sort(([a], [b]) => (a < b ? -1 : 1));
  const text = new TextEncoder().encode(JSON.stringify({ versions, source: describe, options: optionEntries }));
  const digests = await Promise.all(parts);
  const combined = new Uint8Array(text.length + digests.length * 32);
  combined.set(text);
  digests.forEach((d, i) => combined.set(new Uint8Array(d), text.length + i * 32));
  const hash = new Uint8Array(await subtle.digest('SHA-256', combined));
  return `${versions}:sha256:${Array.from(hash, (b) => b.toString(16).padStart(2, '0')).join('')}`;
}

// ------------------------------------------------------------------ IndexedDB

function hasIndexedDb() {
  try {
    return typeof indexedDB !== 'undefined' && indexedDB !== null;
  } catch {
    return false; // some sandboxed iframes throw on access
  }
}

let dbPromise: Promise<IDBDatabase | null> | null = null;

/** Resolves to null instead of rejecting: no database just means no cache. */
function openDb(): Promise<IDBDatabase | null> {
  dbPromise ??= new Promise<IDBDatabase | null>((resolve) => {
    const timer = setTimeout(() => resolve(null), OPEN_TIMEOUT_MS);
    const done = (db: IDBDatabase | null) => {
      clearTimeout(timer);
      resolve(db);
    };
    try {
      const request = indexedDB.open(DB_NAME, 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(DATA_STORE)) db.createObjectStore(DATA_STORE);
        if (!db.objectStoreNames.contains(META_STORE)) db.createObjectStore(META_STORE);
      };
      request.onsuccess = () => {
        const db = request.result;
        // Another tab deleting or upgrading the database: let it, and stop caching in this one.
        db.onversionchange = () => {
          db.close();
          dbPromise = Promise.resolve(null);
        };
        done(db);
      };
      request.onerror = request.onblocked = () => done(null);
    } catch {
      done(null);
    }
  });
  return dbPromise;
}

function readEntry(db: IDBDatabase, key: string): Promise<Uint8Array | null> {
  return new Promise((resolve) => {
    try {
      const request = db.transaction(DATA_STORE, 'readonly').objectStore(DATA_STORE).get(key);
      request.onsuccess = () => {
        const value = request.result;
        resolve(value instanceof ArrayBuffer ? new Uint8Array(value) : value instanceof Uint8Array ? value : null);
      };
      request.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

function writeEntry(db: IDBDatabase, key: string, bytes: Uint8Array, maxBytes = MAX_CACHE_BYTES): Promise<void> {
  return new Promise<void>((resolve) => {
    try {
      const tx = db.transaction([DATA_STORE, META_STORE], 'readwrite');
      tx.objectStore(DATA_STORE).put(bytes.buffer.byteLength === bytes.length ? bytes.buffer : bytes.slice().buffer, key);
      tx.objectStore(META_STORE).put({ size: bytes.length, lastUsed: Date.now() } satisfies MetaEntry, key);
      tx.oncomplete = () => resolve();
      tx.onerror = tx.onabort = () => resolve(); // e.g. QuotaExceededError
    } catch {
      resolve();
    }
  }).then(() => evict(db, key, maxBytes));
}

function touchEntry(db: IDBDatabase, key: string) {
  try {
    const tx = db.transaction(META_STORE, 'readwrite');
    const store = tx.objectStore(META_STORE);
    const request = store.get(key);
    request.onsuccess = () => {
      const meta = request.result as MetaEntry | undefined;
      if (meta) store.put({ ...meta, lastUsed: Date.now() }, key);
    };
  } catch {
    // only affects eviction order
  }
}

function deleteEntry(db: IDBDatabase, key: string) {
  try {
    const tx = db.transaction([DATA_STORE, META_STORE], 'readwrite');
    tx.objectStore(DATA_STORE).delete(key);
    tx.objectStore(META_STORE).delete(key);
  } catch {
    // a stale entry costs disk space only
  }
}

/** Drops least recently used entries (never `keep`) while the cache is over `maxBytes`. */
function evict(db: IDBDatabase, keep: string, maxBytes: number): Promise<void> {
  return new Promise<void>((resolve) => {
    try {
      const tx = db.transaction([DATA_STORE, META_STORE], 'readwrite');
      const meta = tx.objectStore(META_STORE);
      const keysRequest = meta.getAllKeys();
      const valuesRequest = meta.getAll();
      valuesRequest.onsuccess = () => {
        const keys = keysRequest.result as string[];
        const entries = (valuesRequest.result as MetaEntry[]).map((m, i) => ({ key: keys[i], ...m }));
        let total = entries.reduce((acc, e) => acc + e.size, 0);
        entries.sort((a, b) => a.lastUsed - b.lastUsed);
        for (const e of entries) {
          if (total <= maxBytes) break;
          if (e.key === keep) continue;
          tx.objectStore(DATA_STORE).delete(e.key);
          meta.delete(e.key);
          total -= e.size;
        }
      };
      tx.oncomplete = tx.onerror = tx.onabort = () => resolve();
    } catch {
      resolve();
    }
  });
}
