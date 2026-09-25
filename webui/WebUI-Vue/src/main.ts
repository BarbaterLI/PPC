import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
// 拉取 WinUI 库样式令牌与图标字体（副作用导入）。
import 'winui-web-library'
// 显式引入库抽出的全局样式（令牌/动效/图标字体），库在 lib 模式下不会自动注入 CSS。
import 'winui-web-library/dist/winui-web-library.css'
import { createI18n, i18nKey, normalizeLocale } from 'winui-web-library'
import './styles/main.css'
import { THEME_STORAGE_KEY } from './theme/useTheme'

// 挂载前读取持久化主题，为 <html> 加上 theme-light/theme-dark class，
// 保证首帧即为正确明暗令牌。
function readStoredMode(): 'light' | 'dark' {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY)
    return stored === 'dark' || stored === 'high-contrast' ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}
const mode = readStoredMode()
document.documentElement.classList.toggle('theme-dark', mode === 'dark')
document.documentElement.classList.toggle('theme-light', mode === 'light')
document.documentElement.style.colorScheme = mode === 'dark' ? 'dark' : 'light'

const app = createApp(App)
app.provide(i18nKey, createI18n(normalizeLocale(navigator.language)))
app.use(router)
app.mount('#app')