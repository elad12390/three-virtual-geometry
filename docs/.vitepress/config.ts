import { defineConfig } from 'vitepress';

// Served from GitHub Pages at https://elad12390.github.io/three-virtual-geometry/
export default defineConfig({
  title: 'three-virtual-geometry',
  description: 'Virtual geometry for three.js WebGPU: billions of triangles, automatic LOD, one line of code.',
  base: '/three-virtual-geometry/',
  cleanUrls: true,
  lastUpdated: true,
  // The demo is a separate Vite app copied into /demo/ at deploy time and embedded in /live by DemoPlayer.vue.
  ignoreDeadLinks: [/\/demo\//],
  head: [
    ['link', { rel: 'icon', href: '/three-virtual-geometry/favicon.svg', type: 'image/svg+xml' }],
    ['meta', { name: 'theme-color', content: '#101011' }],
    ['meta', { property: 'og:image', content: 'https://elad12390.github.io/three-virtual-geometry/screenshots/ruins-wide.jpg' }],
  ],
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'API', link: '/api' },
      { text: 'Live demo', link: '/live' },
    ],
    sidebar: [
      {
        text: 'Guide',
        items: [
          { text: 'Getting started', link: '/guide/getting-started' },
          { text: 'Importing models (glTF)', link: '/guide/import' },
          { text: 'Meshes and instances', link: '/guide/meshes' },
          { text: 'Settings and levers', link: '/guide/settings' },
          { text: 'Baking and caching', link: '/guide/caching' },
          { text: 'Set it up with your AI agent', link: '/guide/ai-agents' },
        ],
      },
      {
        text: 'Deep dive',
        items: [
          { text: 'How it works', link: '/guide/how-it-works' },
          { text: 'Performance', link: '/guide/performance' },
          { text: 'Troubleshooting', link: '/guide/troubleshooting' },
        ],
      },
      { text: 'API reference', link: '/api' },
    ],
    socialLinks: [{ icon: 'github', link: 'https://github.com/elad12390/three-virtual-geometry' }],
    search: { provider: 'local' },
    editLink: { pattern: 'https://github.com/elad12390/three-virtual-geometry/edit/main/docs/:path' },
    footer: {
      message:
        'MIT License. An independent open-source implementation of virtual geometry, not affiliated with or endorsed by Epic Games.',
      copyright: 'Copyright © 2026 Elad Ben-Haim',
    },
  },
});
