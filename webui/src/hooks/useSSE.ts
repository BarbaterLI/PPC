import { useState, useRef, useCallback } from 'react'

type SSEHookOptions = {
  onMessage?: (event: MessageEvent) => void
  onError?: (error: Event) => void
  onOpen?: () => void
}

export function useSSE<T = any>(options?: SSEHookOptions): {
  data: T | null
  error: Error | null
  isConnected: boolean
  connect: (url: string, eventListeners?: Record<string, (event: MessageEvent) => void>) => EventSource | null
  close: () => void
  eventSource: EventSource | null
} {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [isConnected, setIsConnected] = useState(false)

  const eventSourceRef = useRef<EventSource | null>(null)
  const eventListenersRef = useRef<Record<string, (event: MessageEvent) => void> | null>(null)
  const optionsRef = useRef(options)
  optionsRef.current = options

  const close = useCallback(() => {
    if (eventSourceRef.current) {
      if (eventListenersRef.current) {
        Object.entries(eventListenersRef.current).forEach(([eventName, handler]) => {
          eventSourceRef.current!.removeEventListener(eventName, handler)
        })
        eventListenersRef.current = null
      }
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    setIsConnected(false)
  }, [])

  const connect = useCallback((url: string, eventListeners?: Record<string, (event: MessageEvent) => void>): EventSource | null => {
    close()

    if (!url) {
      return null
    }

    try {
      const es = new EventSource(url)
      eventSourceRef.current = es

      es.onopen = () => {
        setIsConnected(true)
        setError(null)
        optionsRef.current?.onOpen?.()
      }

      es.onmessage = (event) => {
        setError(null)
        try {
          const parsedData = JSON.parse(event.data) as T
          setData(parsedData)
          optionsRef.current?.onMessage?.(event)
        } catch (parseError) {
          console.warn('Failed to parse SSE message data', parseError)
        }
      }

      es.onerror = (event) => {
        setIsConnected(false)
        const err = new Error('SSE connection error')
        setError(err)
        optionsRef.current?.onError?.(event)
      }

      if (eventListeners) {
        Object.entries(eventListeners).forEach(([eventName, handler]) => {
          es.addEventListener(eventName, handler)
        })
        eventListenersRef.current = eventListeners
      }

      return es
    } catch (err) {
      const connectionError = err instanceof Error ? err : new Error('Failed to create SSE connection')
      setError(connectionError)
      setIsConnected(false)
      return null
    }
  }, [close])

  return { data, error, isConnected, connect, close, eventSource: eventSourceRef.current }
}
