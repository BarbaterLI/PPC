import {
  createLightTheme,
  createDarkTheme,
  type Theme,
} from '@fluentui/react-components'
import type { BrandVariants } from '@fluentui/react-components'

const premiumPurple: BrandVariants = {
  10: '#020106',
  20: '#0C0720',
  30: '#140B34',
  40: '#1B0F48',
  50: '#22135D',
  60: '#281873',
  70: '#2E1D89',
  80: '#3423A0',
  90: '#3A29B6',
  100: '#432FCC',
  110: '#5241D3',
  120: '#6254D9',
  130: '#7467DF',
  140: '#867BE5',
  150: '#9A8FEB',
  160: '#AEA4F1',
}

const highContrastBrand: BrandVariants = {
  10: '#000000',
  20: '#0A0A0A',
  30: '#141414',
  40: '#1E1E1E',
  50: '#282828',
  60: '#323232',
  70: '#3C3C3C',
  80: '#464646',
  90: '#505050',
  100: '#5A5A5A',
  110: '#646464',
  120: '#6E6E6E',
  130: '#787878',
  140: '#828282',
  150: '#8C8C8C',
  160: '#969696',
}

const baseLight = createLightTheme(premiumPurple)
const baseDark = createDarkTheme(premiumPurple)

export const lightTheme: Theme = {
  ...baseLight,
  colorNeutralBackground1: '#fafbfc',
  colorNeutralBackground2: '#f4f6f8',
  colorNeutralBackground3: '#eef1f5',
  colorNeutralStroke1: '#e4e8ee',
  colorNeutralStroke2: '#d7dde6',
}

export const darkTheme: Theme = {
  ...baseDark,
  colorNeutralBackground1: '#0d1117',
  colorNeutralBackground2: '#161b22',
  colorNeutralBackground3: '#21262d',
  colorNeutralStroke1: '#30363d',
  colorNeutralStroke2: '#484f58',
}

export const highContrastTheme: Theme = createDarkTheme(highContrastBrand)

export type ThemeMode = 'light' | 'dark' | 'high-contrast'

export const THEME_STORAGE_KEY = 'ppc10-theme-mode'