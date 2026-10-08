import * as THREE from 'three/webgpu';
import { DemoApp } from './demo/app';
import { createMapScene } from './demo/mapScene';
import { createTestScene } from './demo/testScene';
import { createStressScene } from './demo/stressScene';
import { createWorldScene } from './demo/worldScene';
import { createRuinsScene } from './demo/ruinsScene';
import { createImportScene } from './demo/importScene';
import { createScansScene } from './demo/scansScene';
import { createMuseumScene } from './demo/museumScene';
import { runBenchmark } from './demo/bench';

async function main() {
  const app = new DemoApp();
  try {
    await app.init();
    const sceneName = new URLSearchParams(location.search).get('scene') ?? 'map';
    const labels: Record<string, string> = {
      map: '<b>Valley</b> · 2 km map',
      world: '<b>World</b> · 6 km map',
      ruins: '<b>Ruins</b> · showcase',
      stress: '<b>Stress test</b>',
      test: '<b>Test scene</b>',
      import: '<b>glTF import</b> · one call',
      scans: '<b>Scans</b> · real photogrammetry',
      museum: '<b>Museum</b> · architecture',
    };
    app.sceneLabel = labels[sceneName] ?? labels.map;
    if (sceneName === 'test') await createTestScene(app);
    else if (sceneName === 'stress') await createStressScene(app);
    else if (sceneName === 'world') await createWorldScene(app);
    else if (sceneName === 'ruins') await createRuinsScene(app);
    else if (sceneName === 'import') await createImportScene(app);
    else if (sceneName === 'scans') await createScansScene(app);
    else if (sceneName === 'museum') await createMuseumScene(app);
    else await createMapScene(app);
    // Compile every pipeline behind the loading screen: no first-frame freeze, no pop-in.
    app.progress('Compiling shaders…', 1);
    await app.vg.compileAsync(app.renderer, app.scene, app.camera);
    app.hideOverlay();
    app.start();
    Object.assign(window, { app, THREE }); // for debugging from the console
    if (new URLSearchParams(location.search).has('bench')) await runBenchmark(app, sceneName);
  } catch (e) {
    console.error(e);
    app.progress(`Error: ${(e as Error).message}`, 0);
  }
}

main();
