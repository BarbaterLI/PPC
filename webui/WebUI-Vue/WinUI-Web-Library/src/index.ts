// Style entry — tokens, motion and icon font ship with the library bundle.
import './styles/theme.css'
import './styles/animations.css'
import './styles/font.css'

// Style tokens exported as of now (pure CSS, re-exported for discoverability).
export { componentResources, createI18n, i18nKey, normalizeLocale, useI18n } from './components/i18n'

// Public component API: `import { WinButton } from 'winui-web-library'`
export * from './components'