/**
 * useApi —— 复刻 React `src/hooks/useApi.ts`。提供 data/loading/error、refetch、可选轮询、abort。
 */
import { onBeforeUnmount, onMounted, ref, watch, type Ref } from 'vue'
import { request, ApiError } from '@/api/client'
import type { ApiResponse } from '@/api/types'

export type UseApiSource<T> = string | (() => Promise<ApiResponse<T>>)

export interface UseApiOptions {
  refreshInterval?: number
  immediate?: boolean
}

export interface UseApiResult<T> {
  data: Ref<T | null>
  loading: Ref<boolean>
  error: Ref<string | null>
  refetch: () => void
}

function isApiResponse(obj: unknown): obj is ApiResponse {
  return obj !== null && typeof obj === 'object' && 'success' in (obj as Record<string, unknown>)
}

export function useApi<T>(
  source: UseApiSource<T>,
  options: UseApiOptions = {},
): UseApiResult<T> {
  const { refreshInterval, immediate = true } = options

  const data = ref<T | null>(null) as Ref<T | null>
  const loading = ref<boolean>(immediate)
  const error = ref<string | null>(null)

  // 始终持有最新 source（字符串或函数），供轮询取用时引用。
  const sourceHolder = ref<UseApiSource<T>>(source)
  watch(
    () => source,
    (v) => {
      sourceHolder.value = v
    },
    { flush: 'sync' },
  )

  let controller: AbortController | null = null
  let timer: ReturnType<typeof setInterval> | null = null
  let isFirstLoad = true

  const fetchData = async () => {
    controller?.abort()
    const ctrl = new AbortController()
    controller = ctrl

    if (isFirstLoad) {
      loading.value = true
      error.value = null
    }

    try {
      const src = sourceHolder.value
      let result: unknown
      if (typeof src === 'function') {
        result = await src()
      } else {
        result = await request<unknown>(src, {}, ctrl.signal)
      }

      if (ctrl.signal.aborted) return

      if (isApiResponse(result)) {
        if (result.success === false) {
          error.value = result.error?.message || '请求失败'
        } else {
          data.value = result.data as T
          error.value = null
        }
      } else {
        data.value = result as T
        error.value = null
      }

      isFirstLoad = false
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return
      }
      if (!ctrl.signal.aborted) {
        error.value =
          err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : '未知错误'
      }
    } finally {
      loading.value = false
    }
  }

  const refetch = () => {
    void fetchData()
  }

  onMounted(() => {
    if (!immediate) return
    void fetchData()
    if (refreshInterval && refreshInterval > 0) {
      timer = setInterval(() => void fetchData(), refreshInterval)
    }
  })

  onBeforeUnmount(() => {
    controller?.abort()
    if (timer) clearInterval(timer)
  })

  return { data, loading, error, refetch }
}