/**
 * useSse —— 使用 fetch + ReadableStream 消费 SSE，鉴权凭据走 Authorization 头。
 *
 * 不再使用 EventSource 的 ``access_token`` 查询参数：把 token 拼进 URL 会落入
 * 浏览器历史、服务端/代理访问日志与 Referer，属于凭据泄露（OWASP 敏感数据暴露）。
 * fetch 流式可携带请求头，凭据只会出现在请求头中。
 */
import { onBeforeUnmount, ref } from 'vue'
import { getApiToken } from '@/api/client'

const MAX_RECONNECT_ATTEMPTS = 5

export type SSEEventListener = (event: MessageEvent) => void
export type SSEEventListeners = Record<string, SSEEventListener>

export interface UseSSEResult {
  /** 建立连接（可带事件监听表） */
  connect: (url: string, eventListeners?: SSEEventListeners) => void
  /** 主动关闭并重置重连计数 */
  close: () => void
  /** 当前连接状态 */
  active: import('vue').Ref<boolean>
}

export function useSse(): UseSSEResult {
  let controller: AbortController | null = null
  let eventListeners: SSEEventListeners | null = null
  let currentUrl = ''
  let reconnectAttempts = 0
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null

  const active = ref(false)

  const clearReconnectTimer = () => {
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
  }

  const dispatchEvent = (eventName: string, data: string) => {
    if (!eventListeners) return
    const handler = eventListeners[eventName]
    if (!handler) return
    // 与 MessageEvent 形状兼容；消费端仅使用 event.data。
    handler({ data, event: eventName } as unknown as MessageEvent)
  }

  /** 解析一段以 \n\n 结尾的 SSE 块，返回事件名与数据。 */
  const parseBlock = (block: string): { event: string; data: string | null } => {
    let event = 'message'
    let data: string | null = null
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) {
        event = line.slice(6).trim()
      } else if (line.startsWith('data:')) {
        data = line.slice(5).trimStart()
      }
    }
    return { event, data }
  }

  const handleDisconnect = () => {
    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
      const delay = Math.min(1000 * 2 ** reconnectAttempts, 30000)
      reconnectAttempts += 1
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null
        if (currentUrl) {
          void connectInternal(currentUrl)
        }
      }, delay)
    }
  }

  const connectInternal = async (url: string): Promise<void> => {
    clearReconnectTimer()
    controller?.abort()
    controller = new AbortController()

    const token = getApiToken()
    const headers: Record<string, string> = {}
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }

    try {
      const response = await fetch(url, { headers, signal: controller.signal })
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      if (!response.body) {
        throw new Error('No response body')
      }

      active.value = true
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        let blockEnd = buffer.indexOf('\n\n')
        while (blockEnd >= 0) {
          const block = buffer.slice(0, blockEnd)
          buffer = buffer.slice(blockEnd + 2)
          const { event, data } = parseBlock(block)
          if (data !== null) {
            dispatchEvent(event, data)
          }
          blockEnd = buffer.indexOf('\n\n')
        }
      }

      // 服务端正常关闭流（可能已有终态事件；消费端会在 handler 内 close()）。
      active.value = false
      if (!controller.signal.aborted) {
        handleDisconnect()
      }
    } catch (err: unknown) {
      if (controller.signal.aborted || (err as Error)?.name === 'AbortError') {
        // 主动关闭（close/重连），不触发重连。
        active.value = false
        return
      }
      active.value = false
      handleDisconnect()
    }
  }

  const close = () => {
    clearReconnectTimer()
    if (controller) {
      controller.abort()
      controller = null
    }
    eventListeners = null
    reconnectAttempts = 0
    active.value = false
  }

  const connect = (url: string, eventListenersOption?: SSEEventListeners) => {
    currentUrl = url
    if (eventListenersOption) {
      eventListeners = eventListenersOption
    }
    reconnectAttempts = 0
    void connectInternal(url)
  }

  onBeforeUnmount(() => {
    clearReconnectTimer()
    if (controller) {
      controller.abort()
      controller = null
    }
    eventListeners = null
    reconnectAttempts = 0
    active.value = false
  })

  return { connect, close, active }
}