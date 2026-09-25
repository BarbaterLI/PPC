/**
 * 语义化设计令牌 —— 将 React `src/theme/tokens.ts` 语义映射到 WinUI 库的 CSS 变量。
 *
 * 库通过 `import 'winui-web-library'` 拉入 theme.css，其中的 CSS 变量（例如
 * `--app-bg`、`--text-primary`、`--card-stroke`）会随 `<html>` 上的
 * `theme-light`/`theme-dark` class 自动反色。此处仅收敛语义命名与少量纯几何值，
 * 供组件在模板内联样式或 scoped 样式中引用。
 */
export const designTokens = {
  /** 间距：统一页面/容器留白节奏（px） */
  spacing: {
    pageHorizontal: 24,
    pageVertical: 24,
    containerGap: 16,
    sectionGap: 16,
    sectionGapXXL: 32,
  },

  /** 圆角：统一各表面弧度（px） */
  radius: {
    small: 4,
    medium: 6,
    large: 9,
    xlarge: 12,
    circular: 9999,
  },

  /** 表面层级（引用库 CSS 变量，随主题反色） */
  surface: {
    surface1: 'var(--app-bg)',
    surface2: 'var(--layer-default)',
    surface3: 'var(--card-bg)',
    stroke1: 'var(--ctrl-border)',
    stroke2: 'var(--ctrl-border-accent)',
  },

  /** 强调色：品牌主色与前景（引用库 accent 变量） */
  emphasis: {
    brand: 'var(--accent-base)',
    accent: 'var(--accent-text)',
    brandActiveBackground: 'var(--layer-default)',
    brandActiveForeground: 'var(--accent-base)',
  },

  /** 排版字号阶梯（px） */
  typography: {
    headline: 42,
    title: 28,
    subtitle: 18,
    body: 14,
    caption: 12,
    annotation: 11,
  },

  /** 状态语义色（引用 main.css 中的 --ppc-status-* 变量，随明暗主题反色） */
  status: {
    success: 'var(--ppc-status-success, #0f7b0f)',
    warning: 'var(--ppc-status-warning, #9d5d00)',
    danger: 'var(--ppc-status-danger, #c42b1c)',
  },

  /** 动效：时长与缓动（库已提供 CSS 变量，这里仅给常量参考） */
  motion: {
    fast: '167ms',
    normal: '250ms',
    normalSlow: '300ms',
    slow: '500ms',
    fastTransition: 'cubic-bezier(0.0, 0.9, 0.1, 1.0)',
    entrance: 'cubic-bezier(0.1, 0.9, 0.2, 1.0)',
    smooth: 'cubic-bezier(0.33, 0.0, 0.67, 1.0)',
    linear: 'linear',
  },

  /** 布局 */
  layout: {
    /** 内容主区最大宽度（业务内容舒适阅读宽度） */
    pageMaxWidth: '1200px',
    pageHorizontal: 24,
    pageVertical: 24,
  },
} as const