/**
 * Toast 上下文 —— 由 ToastProvider.vue 提供，供任意组件注入使用。
 * 复刻 React `src/hooks/useToast.ts`。
 */
import { inject } from 'vue'
import type { InjectionKey } from 'vue'

export type ToastIntentType = 'success' | 'error' | 'warning' | 'info'

export interface ToastOptions {
  title: string
  body?: string
  intent?: ToastIntentType
  action?: {
    label: string
    onClick: () => void
  }
  duration?: number
}

export interface ToastContextValue {
  showToast: (options: ToastOptions) => void
}

/** 全局 toast 注入键 */
export const toastKey: InjectionKey<ToastContextValue> = Symbol('ppc-toast')

export function useToast(): ToastContextValue {
  const context = inject(toastKey, null)
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return context
}