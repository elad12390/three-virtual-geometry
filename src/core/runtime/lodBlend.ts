/**
 * LOD blending: continuous transitions between the levels of the cluster DAG.
 *
 * A plain cut draws a cluster when `error <= threshold < parentError`, so a cluster switches to its parent at one
 * exact distance, all of its pixels at once: a pop. With a blend band [t, t * blend], every pixel instead shows the
 * cut for its own threshold inside the band, picked by a fixed per-pixel value n in [0, 1): s(n) = t * blend^n.
 * The selection draws every cluster that is in the cut for some threshold in the band, and each one tells the
 * fragment shader the range of n it owns: [f(error), f(parentError)) with f(e) = log(e / t) / log(blend). A pixel
 * keeps a fragment when its n falls in that range, so it shows exactly the cut at s(n), which is complete (no holes)
 * and free of overlaps like any cut. As the camera moves, f changes continuously, so detail moves from one level
 * to the next a few pixels at a time instead of a cluster at once.
 *
 * The ranges are quantized to LOD_FADE_STEPS steps and passed per drawn cluster; a cluster's upper bound and its
 * parent's lower bound come from the same sphere and error, so they are equal bit for bit and neighbouring levels
 * tile the steps exactly.
 */
import { float, floor, fract, screenCoordinate, uniform, varyingProperty } from 'three/tsl';

type Node = any; // eslint-disable-line @typescript-eslint/no-explicit-any

/** Steps of a cluster's blend range: it draws the pixels whose step s is in [lo, hi) of 0..LOD_FADE_STEPS - 1. */
export const LOD_FADE_STEPS = 255;
/** Packed blend range of a cluster drawn over every pixel (lo = 0, hi = LOD_FADE_STEPS), shifted as in the draw list. */
export const FULL_FADE_BITS = LOD_FADE_STEPS << 16;

/** (lo, hi) blend range of the current cluster, written by the vertex stage. */
export const vgFadeVarying = varyingProperty('vec2', 'vVgFade');

/** Frame number for an animated pattern (0: fixed). Set by VirtualGeometry from `lodBlendTemporal`. */
export const lodBlendFrame = uniform(0);

/**
 * True when the current fragment belongs to its pixel's cut. The per-pixel value is interleaved gradient noise
 * (Jimenez 2014): evenly spread over a few pixels, so a half-blended surface looks like a fine, even stipple,
 * and suited to temporal anti-aliasing when it moves every frame.
 */
export function lodBlendMask(): Node {
  const p: Node = screenCoordinate;
  const x = p.x.add(float(lodBlendFrame).mul(5.588238));
  const n = fract(fract(x.mul(0.06711056).add(p.y.mul(0.00583715))).mul(52.9829189));
  const step = floor(n.mul(LOD_FADE_STEPS));
  // The varying is the same at every corner of a triangle; rounding removes interpolation error.
  const range: Node = vgFadeVarying.round();
  return step.greaterThanEqual(range.x).and(step.lessThan(range.y));
}

/**
 * Blend range of a cluster (CPU reference of the GPU selection): its projected own and parent errors in pixels,
 * the error threshold and the band factor. Null when the cluster is in no cut of the band [threshold,
 * threshold * blend] (or its range rounds to nothing). With blend 1 this is the plain cut.
 */
export function lodFadeRange(own: number, parent: number, threshold: number, blend: number): [number, number] | null {
  const high = blend > 1 ? threshold * blend : threshold;
  if (!(own <= high && parent > threshold)) return null;
  if (!(blend > 1)) return [0, LOD_FADE_STEPS];
  const scale = LOD_FADE_STEPS / Math.log(blend);
  const step = (e: number) => Math.min(LOD_FADE_STEPS, Math.max(0, Math.round(Math.log(Math.max(e, 1e-30) / threshold) * scale)));
  const lo = step(own);
  const hi = Number.isFinite(parent) ? step(parent) : LOD_FADE_STEPS;
  return hi > lo ? [lo, hi] : null;
}
