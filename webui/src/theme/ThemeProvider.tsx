import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from 'react'
import { FluentProvider } from '@fluentui/react-components'
import { lightTheme, darkTheme, highContrastTheme, type ThemeMode, THEME_STORAGE_KEY } from './fluentTheme'

interface ThemeContextValue {
  themeMode: ThemeMode
  toggleTheme: () => void
  setThemeMode: (mode: ThemeMode) => void
}

const ThemeContext = createContext<ThemeContextValue | null>(null)

const themeMap: Record<ThemeMode, typeof lightTheme> = {
  light: lightTheme,
  dark: darkTheme,
  'high-contrast': highContrastTheme,
}

const themeOrder: ThemeMode[] = ['light', 'dark', 'high-contrast']

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [themeMode, setThemeModeState] = useState<ThemeMode>(() => {
    try {
      const saved = localStorage.getItem(THEME_STORAGE_KEY)
      if (saved && themeMap[saved as ThemeMode]) return saved as ThemeMode
    } catch (error) {
      console.warn('Theme initialization failed', error)
    }
    return 'light'
  })

  useEffect(() => {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, themeMode)
    } catch (error) {
      console.warn('Failed to persist theme preference', error)
    }
  }, [themeMode])

  const setThemeMode = useCallback((mode: ThemeMode) => {
    setThemeModeState(mode)
  }, [])

  const toggleTheme = useCallback(() => {
    setThemeModeState((prev) => {
      const idx = themeOrder.indexOf(prev)
      return themeOrder[(idx + 1) % themeOrder.length] ?? 'light'
    })
  }, [])

  const currentTheme = themeMap[themeMode]

  return (
    <ThemeContext.Provider value={{ themeMode, toggleTheme, setThemeMode }}>
      <FluentProvider theme={currentTheme}>
        {children}
      </FluentProvider>
    </ThemeContext.Provider>
  )
}

export function useThemeStore(): ThemeContextValue {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useThemeStore must be used within a ThemeProvider')
  }
  return context
}
