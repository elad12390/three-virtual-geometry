import DefaultTheme from 'vitepress/theme';
import type { Theme } from 'vitepress';
import { h } from 'vue';
import DemoPlayer from './DemoPlayer.vue';
import BenchTable from './BenchTable.vue';
import HeroStats from './HeroStats.vue';
import './custom.css';

export default {
  extends: DefaultTheme,
  // Benchmark headlines directly under the home page hero.
  Layout: () => h(DefaultTheme.Layout, null, { 'home-hero-after': () => h(HeroStats) }),
  enhanceApp({ app }) {
    app.component('DemoPlayer', DemoPlayer);
    app.component('BenchTable', BenchTable);
  },
} satisfies Theme;
