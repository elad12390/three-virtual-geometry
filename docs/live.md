---
layout: page
sidebar: false
title: Live demo
---

<div class="live-page">

# Live demo

Every scene runs right here in the page, in your browser, with WebGPU (Chrome, Edge or Safari 26+). Pick a scene
below. Only the selected one runs, and it pauses when you scroll away. Drag to orbit, right-drag to pan and scroll to
zoom. The left panel shows what the GPU is doing; the settings panel on the right holds the levers: detail
threshold, budget and debug views (try **view: meshlets**).

<DemoPlayer autoplay />

The first time a scene loads, its cluster hierarchies are built in the browser (up to about 35 s for the ruins).
Later visits load them from the browser cache in a few seconds.
[Getting started](/guide/getting-started) shows how to do the same with your own models in a few lines.

</div>
