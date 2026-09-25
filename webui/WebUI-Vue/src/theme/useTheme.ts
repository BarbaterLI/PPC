/**
 * 主题 composable —— 通过切换 `<html>` 上的 `theme-light`/`theme-dark` class 驱动 WinUI 库令牌。
 * 复刻 React `src/hooks/useThemeStore.ts` + `src/theme/fluentTheme.ts` 的主题模式语义。
 */
import { ref } from 'vue'

export type ThemeMode = 'light' | 'dark' | 'high-contrast'

export const THEME_STORAGE_KEY = 'ppc-theme-mode'

function isThemeMode(value: string | null): value is ThemeMode {
  return value === 'light' || value === 'dark' || value === 'high-contrast'
}

function initialMode(): ThemeMode {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY)
    if (isThemeMode(stored)) return stored
  } catch {
    /* ignore */
  }
  return 'light'
}

function applyMode(mode: ThemeMode): void {
  const el = document.documentElement
  el.classList.toggle('theme-dark', mode !== 'light')
  el.classList.toggle('theme-light', mode === 'light')
  el.style.colorScheme = mode === 'light' ? 'light' : 'dark'
}

/** 模块级单例状态：保证多组件共享同一主题实例。 */
const mode = ref<ThemeMode>(initialMode())

export function useTheme() {
  /** 主题循环：light → dark → high-contrast → light */
  const CYCLE: ThemeMode[] = ['light', 'dark', 'high-contrast']

  const setMode = (target: ThemeMode): void => {
    mode.value = target
    applyMode(target)
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, target)
    } catch {
      /* ignore */
    }
  }

  const toggleTheme = (): void => {
    const idx = CYCLE.indexOf(mode.value)
    setMode(CYCLE[(idx + 1) % CYCLE.length])
  }

  // 首次调用时应用持久化主题（幂等，重复挂载无害）。
  if (document.readyState !== 'loading') {
    applyMode(mode.value)
  }

  return { mode, setMode, toggleTheme }
}