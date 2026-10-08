import DefaultTheme from 'vitepress/theme';
import type { Theme } from 'vitepress';
import DemoPlayer from './DemoPlayer.vue';
import BenchTable from './BenchTable.vue';
import './custom.css';

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('DemoPlayer', DemoPlayer);
    app.component('BenchTable', BenchTable);
  },
} satisfies Theme;
