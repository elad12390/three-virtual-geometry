import * as THREE from 'three/webgpu';

/** A camera keyframe: look at `target` from `dist` away, at `yaw` around Y (0: from +z) and `pitch` above it. */
export interface TourKey {
  target: [number, number, number];
  dist: number;
  yaw: number;
  pitch: number;
  /** Seconds from the start of the tour. */
  time: number;
}

/**
 * A looping camera fly-through. Monotone cubic (Fritsch-Carlson) interpolation per channel and in time: velocity stays
 * continuous and nothing overshoots, even between a 1 m close-up and a 600 m panorama. Distance is interpolated in log
 * space, so a pull-out reads as one steady zoom. `minHeight(x, z)` keeps the camera above the ground.
 */
export function makeTour(keys: TourKey[], minHeight: (x: number, z: number) => number = () => -Infinity) {
  const loopTime = keys[keys.length - 1].time;
  const channels = keys.map((k) => [k.target[0], k.target[1], k.target[2], Math.log(k.dist), k.yaw, k.pitch]);
  const slopes = keys.map((_, i) =>
    channels[i].map((_, c) => {
      if (i === 0 || i === keys.length - 1) return 0;
      const d0 = (channels[i][c] - channels[i - 1][c]) / (keys[i].time - keys[i - 1].time);
      const d1 = (channels[i + 1][c] - channels[i][c]) / (keys[i + 1].time - keys[i].time);
      if (d0 * d1 <= 0) return 0;
      return Math.sign(d0) * Math.min(Math.abs(d0 + d1) / 2, 3 * Math.min(Math.abs(d0), Math.abs(d1)));
    })
  );
  const pose = new Array<number>(6);
  /** Writes the camera target and position at `time` (seconds, looping). Returns the distance between them. */
  const poseAt = (time: number, target: THREE.Vector3, position: THREE.Vector3) => {
    const t = ((time % loopTime) + loopTime) % loopTime;
    let i = 0;
    while (i < keys.length - 2 && t >= keys[i + 1].time) i++;
    const h = keys[i + 1].time - keys[i].time;
    const u = (t - keys[i].time) / h;
    const u2 = u * u;
    const u3 = u2 * u;
    for (let c = 0; c < 6; c++) {
      pose[c] =
        (2 * u3 - 3 * u2 + 1) * channels[i][c] + (u3 - 2 * u2 + u) * h * slopes[i][c] + (-2 * u3 + 3 * u2) * channels[i + 1][c] + (u3 - u2) * h * slopes[i + 1][c];
    }
    const [tx, ty, tz, logDist, yaw, pitch] = pose;
    const dist = Math.exp(logDist);
    target.set(tx, ty, tz);
    position.set(tx + dist * Math.cos(pitch) * Math.sin(yaw), ty + dist * Math.sin(pitch), tz + dist * Math.cos(pitch) * Math.cos(yaw));
    position.y = Math.max(position.y, minHeight(position.x, position.z));
    return dist;
  };
  return { poseAt, duration: loopTime };
}
