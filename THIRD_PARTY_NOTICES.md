# Third-party notices

## nanite-webgpu

Parts of the mesh preprocessing (`src/nanite/preprocess/buildNaniteMesh.ts`) and of the GPU culling
pipeline (`src/nanite/runtime/NaniteMesh.ts`) were ported from
[nanite-webgpu](https://github.com/Scthe/nanite-webgpu) and have since been substantially rewritten.

```
The MIT License (MIT)

Copyright (c) 2024 Marcin Matuszczyk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Runtime dependencies

`three` and `meshoptimizer` are npm dependencies (both MIT) and are not vendored here; their licenses
ship with their packages.

## Demo assets (not part of the npm package)

`examples/public/models/kenney-car-kit/`: [Car Kit](https://kenney.nl/assets/car-kit) by Kenney (www.kenney.nl),
CC0 1.0 Universal (public domain). See `License.txt` in that folder. All other demo scenes are generated
procedurally in code.
