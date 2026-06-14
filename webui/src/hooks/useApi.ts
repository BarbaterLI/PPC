import { useState, useEffect, useCallback, useRef } from 'react'
import { request, ApiError } from '@/api/client'
import type { ApiResponse } from '@/types'

interface UseApiResult<T> {
  data: T | null
  loading: boolean
  error: string | null
  refetch: () => void
}

interface UseApiOptions {
  refreshInterval?: number
  immediate?: boolean
}

type UseApiSource<T> = string | (() => Promise<ApiResponse<T>>)

function isApiResponse(obj: unknown): obj is ApiResponse {
  return obj !== null && typeof obj === 'object' && 'success' in (obj as Record<string, unknown>)
}

export function useApi<T>(
  source: UseApiSource<T>,
  options: UseApiOptions = {},
): UseApiResult<T> {
  const { refreshInterval, immediate = true } = options

  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState<boolean>(immediate)
  const [error, setError] = useState<string | null>(null)

  const sourceRef = useRef(source)
  sourceRef.current = source

  const abortControllerRef = useRef<AbortController | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const isFirstLoad = useRef(true)

  const sourceKey = typeof source === 'string' ? source : typeof source === 'function' ? 'function-source' : undefined

  const fetchData = useCallback(async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    const controller = new AbortController()
    abortControllerRef.current = controller

    if (isFirstLoad.current) {
      setLoading(true)
      setError(null)
    }

    try {
      const currentSource = sourceRef.current
      let result: unknown

      if (typeof currentSource === 'function') {
        result = await currentSource()
      } else {
        result = await request<unknown>(currentSource, {}, controller.signal)
      }

      if (controller.signal.aborted) return

      if (isApiResponse(result)) {
        if (result.success === false) {
          setError(result.error || result.message || '请求失败')
        } else {
          setData(result.data as T)
          setError(null)
        }
      } else {
        setData(result as T)
        setError(null)
      }

      isFirstLoad.current = false
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return
      }

      if (!controller.signal.aborted) {
        const message =
          err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : '未知错误'
        setError(message)
      }
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false)
      }
    }
  }, [sourceKey])

  useEffect(() => {
    if (!immediate) return

    fetchData()

    if (refreshInterval && refreshInterval > 0) {
      intervalRef.current = setInterval(fetchData, refreshInterval)
    }

    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [fetchData, refreshInterval, immediate])

  const refetch = useCallback(() => {
    fetchData()
  }, [fetchData])

  return { data, loading, error, refetch }
}
