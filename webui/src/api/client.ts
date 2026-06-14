import type {
  ApiResponse,
  SystemStatus,
  PPC10Config,
  ConfigSection,
  ConfigUpdatePayload,
  ConfigExportData,
  Task,
  TaskProgress,
  ConvertTaskParams,
  AnalysisResult,
  Extension,
  ExtensionWebUIConfig,
  FanqieStatus,
  FanqieConfig,
  FanqieInstallParams,
  FanqieLaunchServerParams,
  VoiceGroup,
  SplitParams,
  MergeParams,
  PreviewParams,
  HealthReport,
  ClusterStatus,
  NodeInfo,
  ClusterMetricsResponse,
  TaskAssignment,
} from '../types'

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

async function request<T>(
  url: string,
  options: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> | undefined),
  }

  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
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
    let message = `HTTP ${response.status}`
    let code: string | undefined
    try {
      const body = await response.json()
      message = body.error || body.message || message
      code = body.code
    } catch (error) {
      console.debug('Failed to parse error response body', error)
    }
    throw new ApiError(message, response.status, code)
  }

  if (response.headers.get('content-type')?.includes('application/json')) {
    const data = await response.json()
    return data as T
  }

  throw new Error('Response is not JSON')
}

async function requestBlob(
  url: string,
  options: RequestInit = {},
  signal?: AbortSignal,
): Promise<Blob> {
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> | undefined),
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
    let message = `HTTP ${response.status}`
    try {
      const body = await response.json()
      message = body.error || body.message || message
    } catch {}
    throw new ApiError(message, response.status)
  }

  return response.blob()
}

function wrapApiResponse<T>(data: T): ApiResponse<T> {
  return { success: true, data }
}

function createSSEConnection(
  url: string,
  onMessage: (data: TaskProgress) => void,
  onError?: (error: Event) => void,
): EventSource {
  const source = new EventSource(url)

  source.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as TaskProgress
      onMessage(data)
    } catch (error) {
      console.debug('Failed to parse SSE message data', error)
    }
  }

  source.onerror = (event) => {
    onError?.(event)
  }

  return source
}

export { ApiError, request, requestBlob, createSSEConnection }

export const systemApi = {
  getStatus: () =>
    request<ApiResponse<SystemStatus>>('/api/status'),

  check: () =>
    request<ApiResponse<HealthReport>>('/api/check'),

  getVoices: () =>
    request<ApiResponse<VoiceGroup[]>>('/api/voices'),
}

export const configApi = {
  getAll: () =>
    request<ApiResponse<PPC10Config[]>>('/api/config'),

  getByKey: (key: string) =>
    request<ApiResponse<ConfigSection>>(`/api/config/${encodeURIComponent(key)}`),

  update: (payload: ConfigUpdatePayload) =>
    request<ApiResponse<void>>('/api/config', {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

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

  export: () =>
    request<ApiResponse<ConfigExportData>>('/api/config/export', {
      method: 'POST',
    }),

  exportFile: () =>
    requestBlob('/api/config/export', { method: 'POST' }),

  import: (data: ConfigExportData) =>
    request<ApiResponse<void>>('/api/config/import', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

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
  createConvert: (params: ConvertTaskParams) =>
    request<ApiResponse<{ task_id: string }>>('/api/tasks/convert', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  list: () =>
    request<ApiResponse<Task[]>>('/api/tasks'),

  getById: (id: string) =>
    request<ApiResponse<Task>>(`/api/tasks/${encodeURIComponent(id)}`),

  streamProgress: (id: string) =>
    `/api/tasks/${encodeURIComponent(id)}/stream`,

  cancel: (id: string) =>
    request<ApiResponse<void>>(`/api/tasks/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
}

export const analyzeApi = {
  run: (modules?: string[]) =>
    request<ApiResponse<AnalysisResult>>('/api/analyze', {
      method: 'POST',
      body: JSON.stringify({ modules }),
    }),

  getHistory: () =>
    request<ApiResponse<AnalysisResult[]>>('/api/analyze/history'),
}

export const operationApi = {
  split: (params: SplitParams) =>
    request<ApiResponse<{ output_dir: string; file_count: number }>>('/api/split', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  merge: (params: MergeParams) =>
    request<ApiResponse<{ output_path: string }>>('/api/merge', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  preview: (params: PreviewParams) =>
    request<ApiResponse<{ audio_url: string }>>('/api/preview', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
}

export const extensionApi = {
  list: () =>
    request<ApiResponse<Extension[]>>('/api/extensions'),

  getWebUIConfigs: () =>
    request<ApiResponse<ExtensionWebUIConfig[]>>('/api/extensions/webui'),

  install: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return request<ApiResponse<Extension>>('/api/extensions/install', {
      method: 'POST',
      headers: {},
      body: formData,
    })
  },

  uninstall: (name: string) =>
    request<ApiResponse<void>>(`/api/extensions/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),

  getByName: (name: string) =>
    request<ApiResponse<Extension>>(`/api/extensions/${encodeURIComponent(name)}`),

  enable: (name: string) =>
    request<ApiResponse<void>>(`/api/extensions/${encodeURIComponent(name)}/enable`, {
      method: 'POST',
    }),

  disable: (name: string) =>
    request<ApiResponse<void>>(`/api/extensions/${encodeURIComponent(name)}/disable`, {
      method: 'POST',
    }),
}

export const fanqieApi = {
  getStatus: () =>
    request<ApiResponse<FanqieStatus>>('/api/fanqie/status'),

  install: (params?: FanqieInstallParams) =>
    request<ApiResponse<{ version: string }>>('/api/fanqie/install', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  launchServer: (params?: FanqieLaunchServerParams) =>
    request<ApiResponse<{ host: string; port: number }>>('/api/fanqie/launch-server', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  stopServer: () =>
    request<ApiResponse<void>>('/api/fanqie/stop-server', {
      method: 'POST',
    }),

  getConfig: () =>
    request<ApiResponse<FanqieConfig>>('/api/fanqie/config'),

  updateConfig: (config: Partial<FanqieConfig>) =>
    request<ApiResponse<void>>('/api/fanqie/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    }),

  uninstall: () =>
    request<ApiResponse<void>>('/api/fanqie/uninstall', {
      method: 'POST',
    }),
}

export const distributedApi = {
  getStatus: async () =>
    wrapApiResponse(await request<ClusterStatus>('/api/distributed/status')),
  getNodes: async () =>
    wrapApiResponse(await request<NodeInfo[]>('/api/distributed/nodes')),
  addNode: async (host: string, port: number, maxConcurrency: number) =>
    wrapApiResponse(await request<NodeInfo>('/api/distributed/nodes', { method: 'POST', body: JSON.stringify({ host, port, max_concurrency: maxConcurrency }) })),
  removeNode: async (nodeId: string) =>
    wrapApiResponse(await request<void>(`/api/distributed/nodes/${encodeURIComponent(nodeId)}`, { method: 'DELETE' })),
  drainNode: async (nodeId: string) =>
    wrapApiResponse(await request<NodeInfo>(`/api/distributed/nodes/${encodeURIComponent(nodeId)}/drain`, { method: 'POST' })),
  activateNode: async (nodeId: string) =>
    wrapApiResponse(await request<NodeInfo>(`/api/distributed/nodes/${encodeURIComponent(nodeId)}/activate`, { method: 'POST' })),
  getMetrics: async () =>
    wrapApiResponse(await request<ClusterMetricsResponse>('/api/distributed/metrics')),
  getTasks: async () =>
    wrapApiResponse(await request<TaskAssignment[]>('/api/distributed/tasks')),
  startScheduler: async (strategy = 'round_robin', localExecution = true) =>
    wrapApiResponse(await request<{ status: string }>('/api/distributed/start', { method: 'POST', body: JSON.stringify({ strategy, local_execution: localExecution }) })),
  stopScheduler: async () =>
    wrapApiResponse(await request<{ status: string }>('/api/distributed/stop', { method: 'POST' })),
  startNodeService: async (host = '0.0.0.0', port = 8080, maxConcurrency = 4) =>
    wrapApiResponse(await request<{ status: string }>('/api/distributed/node-service/start', { method: 'POST', body: JSON.stringify({ host, port, max_concurrency: maxConcurrency }) })),
  stopNodeService: async () =>
    wrapApiResponse(await request<{ status: string }>('/api/distributed/node-service/stop', { method: 'POST' })),
}
