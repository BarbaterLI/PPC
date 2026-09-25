/**
 * 响应式断点 —— 复刻 React 工程 `src/constants/breakpoints.ts`。
 */

export const BREAKPOINTS = {
  MOBILE: '@media (max-width: 767px)',
  TABLET: '@media (min-width: 768px) and (max-width: 1023px)',
  DESKTOP: '@media (min-width: 1024px)',
  LARGE: '@media (min-width: 1440px)',
  UP_TO_TABLET_MIN: '@media (max-width: 768px)',
  COMPACT: '@media (max-width: 900px)',
  SMALL: '@media (max-width: 500px)',
} as const

export const MOBILE_MAX_WIDTH = 767
export const TABLET_MIN_WIDTH = 768
export const TABLET_MAX_WIDTH = 1023
export const DESKTOP_MIN_WIDTH = 1024
export const LARGE_MIN_WIDTH = 1440
export const COMPACT_MAX_WIDTH = 900
export const SMALL_MAX_WIDTH = 500