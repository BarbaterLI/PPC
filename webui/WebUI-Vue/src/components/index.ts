/**
 * 共享 UI 组件统一出口 —— 供页面以 `import { PageHeader, EmptyState } from '@/components'`
 * 的方式引用，避免每一处都写又长又易错的相对路径。
 */
export { default as PageHeader } from './PageHeader.vue'
export { default as StatusBadge } from './StatusBadge.vue'
export { default as ConfirmDialog } from './ConfirmDialog.vue'
export { default as EmptyState } from './EmptyState.vue'
export { default as LoadingSkeleton } from './LoadingSkeleton.vue'
export { default as ErrorBoundary } from './ErrorBoundary.vue'
export { default as ToastProvider } from './ToastProvider.vue'
export { default as AppLayout } from './AppLayout.vue'

export type { StatusType, StatusBadgeSize } from './StatusBadge.vue'