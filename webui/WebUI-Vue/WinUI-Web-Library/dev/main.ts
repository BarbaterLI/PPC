import { createApp } from 'vue'
import { createI18n, i18nKey, normalizeLocale } from '../src'
import DemoApp from './DemoApp.vue'

const app = createApp(DemoApp)
app.provide(i18nKey, createI18n(normalizeLocale(navigator.language)))
app.mount('#app')