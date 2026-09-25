/**
 * useApiAction —— 复刻 React `src/hooks/useApiAction.ts`。执行异步 action，
 * 控制 loading，统一 success / error 回调（未提供 onError 时自动弹 toast）。
 */
import { ref, type Ref } from 'vue'
import { useToast } from './useToast'
import type { ToastOptions } from './useToast'

export interface ApiActionOptions {
  successToast?: ToastOptions | (() => ToastOptions)
  errorTitle?: string
  onSuccess?: (result: unknown) => void
  onError?: (message: string) => void
}

export interface UseApiActionResult {
  loading: Ref<boolean>
  run: () => Promise<void>
}

export function useApiAction(
  action: () => Promise<unknown>,
  options: ApiActionOptions = {},
): UseApiActionResult {
  const loading = ref(false)
  const { showToast } = useToast()

  // 始终持有最新 action / options，避免闭包过期。
  const actionRef = { current: action }
  const optionsRef = { current: options }
  let gate = false

  const run = async () => {
    if (gate) return
    gate = true
    loading.value = true

    try {
      const result = await actionRef.current()
      const opts = optionsRef.current
      if (opts.successToast) {
        showToast(typeof opts.successToast === 'function' ? opts.successToast() : opts.successToast)
      }
      opts.onSuccess?.(result)
    } catch (err: unknown) {
      const opts = optionsRef.current
      const message = err instanceof Error ? err.message : '操作失败'
      if (opts.onError) {
        opts.onError(message)
      } else {
        showToast({ title: opts.errorTitle ?? '操作失败', body: message, intent: 'error' })
      }
    } finally {
      gate = false
      loading.value = false
    }
  }

  return { loading, run }
}