/**
 * 路由 —— 5 条核心业务路由 + 1 条扩展动态路由挂在带 AppLayout 的布局路由之下，
 * 页面组件懒加载，路径与 React 工程保持一致（/config、/tts-convert 等）。
 * 扩展 WebUI 通过 /ext/:name 动态路由加载（如 /ext/fanqie）。
 */
import { createRouter, createWebHistory } from 'vue-router'
import AppLayout from './components/AppLayout.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      component: AppLayout,
      children: [
        { path: '', redirect: '/dashboard' },
        {
          path: 'dashboard',
          name: 'dashboard',
          component: () => import('./pages/DashboardPage.vue'),
        },
        {
          path: 'config',
          name: 'config',
          component: () => import('./pages/ConfigPage.vue'),
        },
        {
          path: 'tts-convert',
          name: 'tts-convert',
          component: () => import('./pages/TtsConvertPage.vue'),
        },
        {
          path: 'extensions',
          name: 'extensions',
          component: () => import('./pages/ExtensionsPage.vue'),
        },
        {
          path: 'distributed',
          name: 'distributed',
          component: () => import('./pages/DistributedPage.vue'),
        },
        {
          path: 'ext/:name',
          name: 'ext',
          props: true,
          component: () => import('./pages/ExtensionPage.vue'),
        },
      ],
    },
  ],
})

export default router