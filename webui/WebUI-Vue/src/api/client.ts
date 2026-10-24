/**
 * API 客户端 —— 复刻 React 工程 `src/api/client.ts`。
 * 端点、请求体、Bearer 鉴权、错误处理逻辑保持一致，仅按 Vue/TS 习惯微调签名。
 */
import type {
  ApiResponse,
  SystemStatus,
  PPCConfig,
  ConvertTaskParams,
  Extension,
  ExtensionWebUIConfig,
  VoiceGroup,
  ClusterStatus,
  NodeInfo,
  ClusterMetricsResponse,
  TaskAssignment,
  Task,
  PagedTasks,
  TaskListQuery,
} from './types'

class ApiError extends Error {
  status: number
  code?: string

  constructor(message: string, status: number, code?: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

async function handleErrorResponse(response: Response): Promise<ApiError> {
  let message = `HTTP ${response.status}`
  let code: string | undefined
  try {
    const body = (await response.json()) as ApiResponse<unknown>
    message = body.error?.message || message
    code = body.error?.code
  } catch {
    const text = await response.text().catch(() => '')
    const snippet = text.slice(0, 200)
    return new ApiError(
      `HTTP ${response.status}: ${snippet || 'non-JSON response'}`,
      response.status,
      'PARSE_ERROR',
    )
  }
  return new ApiError(message, response.status, code)
}

// ---- 可选 API Token（Bearer）鉴权 ----
const API_TOKEN_KEY = 'ppc_api_token'

export function getApiToken(): string {
  try {
    return window.localStorage.getItem(API_TOKEN_KEY) || ''
  } catch {
    return ''
  }
}

export function setApiToken(token: string): void {
  try {
    window.localStorage.setItem(API_TOKEN_KEY, token)
  } catch {
    /* ignore */
  }
}

export function clearApiToken(): void {
  try {
    window.localStorage.removeItem(API_TOKEN_KEY)
  } catch {
    /* ignore */
  }
}

async function promptForToken(): Promise<string | undefined> {
  if (typeof window === 'undefined' || typeof window.prompt !== 'function') return undefined
  const input = window.prompt('此服务要求 API Token，请输入（可在服务端配置中找到）')
  if (input == null) return undefined
  const token = input.trim()
  if (token) setApiToken(token)
  return token || undefined
}

async function request<T>(
  url: string,
  options: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  return requestInner<T>(url, options, signal, false)
}

async function requestInner<T>(
  url: string,
  options: RequestInit,
  signal: AbortSignal | undefined,
  retriedAuth: boolean,
): Promise<T> {
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> | undefined),
  }

  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }

  const token = getApiToken()
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  const fetchOptions: RequestInit = {
    ...options,
    headers,
  }
  if (signal !== undefined) {
    fetchOptions.signal = signal
  }

  let response = await fetch(url, fetchOptions)

  if (response.status === 401 && !retriedAuth && !token) {
    const entered = await promptForToken()
    if (entered) {
      // promptForToken 已把新 token 写入 localStorage，重试时基于它重建 headers，
      // 否则复用旧的 fetchOptions（不含 Authorization）会导致重试依然 401。
      const retryHeaders = { ...fetchOptions.headers, Authorization: `Bearer ${getApiToken()}` }
      response = await fetch(url, { ...fetchOptions, headers: retryHeaders })
    }
  }

  if (!response.ok) {
    throw await handleErrorResponse(response)
  }

  if (response.headers.get('content-type')?.includes('application/json')) {
    try {
      const data = await response.json()
      return data as T
    } catch {
      const text = await response.text().catch(() => '')
      const snippet = text.slice(0, 200)
      throw new ApiError(
        `HTTP ${response.status}: ${snippet || 'invalid JSON'}`,
        response.status,
        'PARSE_ERROR',
      )
    }
  }

  throw new ApiError('Response is not JSON', response.status, 'NOT_JSON')
}

async function requestBlob(
  url: string,
  options: RequestInit = {},
  signal?: AbortSignal,
): Promise<Blob> {
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> | undefined),
  }
  const token = getApiToken()
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  const fetchOptions: RequestInit = {
    ...options,
    headers,
  }
  if (signal !== undefined) {
    fetchOptions.signal = signal
  }

  const response = await fetch(url, fetchOptions)

  if (!response.ok) {
    throw await handleErrorResponse(response)
  }

  return response.blob()
}

function uploadWithProgress(
  url: string,
  file: File,
  onProgress?: (percent: number) => void,
  signal?: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    const formData = new FormData()
    formData.append('file', file)

    const onAbort = () => {
      xhr.abort()
    }
    if (signal) {
      if (signal.aborted) {
        onAbort()
      } else {
        signal.addEventListener('abort', onAbort, { once: true })
      }
    }

    const cleanup = () => {
      signal?.removeEventListener('abort', onAbort)
    }

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        onProgress?.(Math.round((e.loaded / e.total) * 100))
      }
    })

    xhr.addEventListener('load', () => {
      cleanup()
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve()
        return
      }
      let message = `请求失败: ${xhr.status}`
      try {
        const body = JSON.parse(xhr.responseText) as ApiResponse<unknown>
        message = body.error?.message || message
      } catch {
        if (xhr.responseText) {
          message = xhr.responseText
        }
      }
      reject(new ApiError(message, xhr.status))
    })

    xhr.addEventListener('error', () => {
      cleanup()
      if (signal?.aborted) {
        reject(new DOMException('Aborted', 'AbortError'))
        return
      }
      reject(new ApiError('网络错误，请重试', 0))
    })

    xhr.open('POST', url)
    const token = getApiToken()
    if (token) {
      xhr.setRequestHeader('Authorization', `Bearer ${token}`)
    }
    xhr.send(formData)
  })
}

export { ApiError, request, requestBlob, uploadWithProgress }

/** 目录浏览会话（只读），供目录选择器便捷定位服务器目录。 */
export interface BrowseDirectory {
  path: string
  parent: string
  home: string
  folders: string[]
  is_root: boolean
}

export const browseApi = {
  list: (path?: string) =>
    request<ApiResponse<BrowseDirectory>>(
      path ? `/api/browse?path=${encodeURIComponent(path)}` : '/api/browse',
    ),
}

export const systemApi = {
  getStatus: () => request<ApiResponse<SystemStatus>>('/api/status'),
  getVoices: () => request<ApiResponse<VoiceGroup[]>>('/api/voices'),
}

export const configApi = {
  getAll: () => request<ApiResponse<PPCConfig[]>>('/api/config'),

  batchUpdate: (payloads: Array<{ key: string; value: string }>) =>
    request<ApiResponse<{ failed?: Array<{ key: string }> }>>('/api/config/batch', {
      method: 'PUT',
      body: JSON.stringify(payloads),
    }),

  reset: (preset?: string) =>
    request<ApiResponse<void>>('/api/config/reset', {
      method: 'POST',
      body: JSON.stringify({ preset }),
    }),

  exportFile: () => requestBlob('/api/config/export/file'),

  importFile: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return request<ApiResponse<void>>('/api/config/import', {
      method: 'POST',
      headers: {},
      body: formData,
    })
  },
}

export const taskApi = {
  list: () => request<ApiResponse<Task[]>>('/api/tasks'),

  listWithQuery: (query: TaskListQuery = {}) => {
    const params = new URLSearchParams()
    if (query.status) params.set('status', query.status)
    if (query.limit !== undefined) params.set('limit', String(query.limit))
    if (query.offset !== undefined) params.set('offset', String(query.offset))
    const qs = params.toString()
    return request<ApiResponse<PagedTasks>>(`/api/tasks${qs ? `?${qs}` : ''}`)
  },

  downloadUrl: (id: string) => `/api/tasks/${encodeURIComponent(id)}/download`,

  createConvert: (params: ConvertTaskParams) =>
    request<ApiResponse<{ task_id: string }>>('/api/tasks/convert', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  streamProgress: (id: string) => `/api/tasks/${encodeURIComponent(id)}/stream`,

  cancel: (id: string) =>
    request<ApiResponse<void>>(`/api/tasks/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
}

export const extensionApi = {
  list: () => request<ApiResponse<Extension[]>>('/api/extensions'),

  getWebUIConfigs: () =>
    request<ApiResponse<ExtensionWebUIConfig[]>>('/api/extensions/webui'),

  installWithProgress: (file: File, onProgress?: (percent: number) => void, signal?: AbortSignal) =>
    uploadWithProgress('/api/extensions/install', file, onProgress, signal),

  uninstall: (name: string) =>
    request<ApiResponse<void>>(`/api/extensions/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),

  enable: (name: string) =>
    request<ApiResponse<void>>(`/api/extensions/${encodeURIComponent(name)}/enable`, {
      method: 'POST',
    }),

  disable: (name: string) =>
    request<ApiResponse<void>>(`/api/extensions/${encodeURIComponent(name)}/disable`, {
      method: 'POST',
    }),
}

export const distributedApi = {
  getStatus: () => request<ApiResponse<ClusterStatus>>('/api/distributed/status'),
  getNodes: () => request<ApiResponse<NodeInfo[]>>('/api/distributed/nodes'),
  addNode: (host: string, port: number, maxConcurrency: number) =>
    request<ApiResponse<NodeInfo>>('/api/distributed/nodes', {
      method: 'POST',
      body: JSON.stringify({ host, port, max_concurrency: maxConcurrency }),
    }),
  removeNode: (nodeId: string) =>
    request<ApiResponse<void>>(`/api/distributed/nodes/${encodeURIComponent(nodeId)}`, {
      method: 'DELETE',
    }),
  drainNode: (nodeId: string) =>
    request<ApiResponse<NodeInfo>>(`/api/distributed/nodes/${encodeURIComponent(nodeId)}/drain`, {
      method: 'POST',
    }),
  activateNode: (nodeId: string) =>
    request<ApiResponse<NodeInfo>>(`/api/distributed/nodes/${encodeURIComponent(nodeId)}/activate`, {
      method: 'POST',
    }),
  getMetrics: () => request<ApiResponse<ClusterMetricsResponse>>('/api/distributed/metrics'),
  getTasks: () => request<ApiResponse<TaskAssignment[]>>('/api/distributed/tasks'),
  startScheduler: (strategy = 'round_robin', localExecution = true) =>
    request<ApiResponse<{ status: string }>>('/api/distributed/start', {
      method: 'POST',
      body: JSON.stringify({ strategy, local_execution: localExecution }),
    }),
  stopScheduler: () =>
    request<ApiResponse<{ status: string }>>('/api/distributed/stop', { method: 'POST' }),
  startNodeService: (host = '0.0.0.0', port = 8080, maxConcurrency = 4) =>
    request<ApiResponse<{ status: string }>>('/api/distributed/node-service/start', {
      method: 'POST',
      body: JSON.stringify({ host, port, max_concurrency: maxConcurrency }),
    }),
  stopNodeService: () =>
    request<ApiResponse<{ status: string }>>('/api/distributed/node-service/stop', { method: 'POST' }),
}