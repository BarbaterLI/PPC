/**
 * API 业务模型类型 —— 复刻 React 工程 `src/types/index.ts`。
 * 所有键名与后端响应保持一致（snake_case）。
 */

export interface SystemStatus {
  status: 'running' | 'stopped' | 'error' | 'unknown'
  uptime: number
  version: string
  cpu_percent: number
  memory_percent: number
  disk_percent: number
  health_score: number
  active_tasks: number
  timestamp: string
}

export type AudioFormat = 'mp3' | 'wav' | 'ogg' | 'aac'
export type AudioQuality = 'low' | 'medium' | 'high' | 'lossless'

export interface ConfigItem {
  key: string
  value: unknown
  type: 'string' | 'number' | 'boolean' | 'array' | 'object'
  description: string
  default_value: unknown
  editable: boolean
}

export interface ConfigSection {
  name: string
  label: string
  description: string
  items: ConfigItem[]
}

export interface ConfigUpdatePayload {
  key: string
  value: unknown
}

export interface ConfigExportData {
  version: string
  exported_at: string
  config: Record<string, unknown>
}

export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface Task {
  task_id: string
  task_type: string
  status: TaskStatus
  created_at: string | number
  started_at?: string | number | null
  completed_at?: string | number | null
  params: Record<string, unknown>
  result?: Record<string, unknown> | null
  error?: string | null
  progress: number
  progress_message?: string
}

export interface PagedTasks {
  items: Task[]
  total: number
}

export interface TaskListQuery {
  status?: TaskStatus
  limit?: number
  offset?: number
}

export interface ConvertTaskParams {
  input_dir: string
  output_dir: string
  voice: string
  concurrency?: number
  format?: AudioFormat
  quality?: AudioQuality
  rate?: string
  tts_backend?: string
  tts_proxy?: string
  segment_bitrate?: string
  recursive?: boolean
  resume?: boolean
}

export type ExtensionType =
  | 'load_balance_strategy'
  | 'health_check_strategy'
  | 'task_scheduling_strategy'
  | 'metrics_exporter'
  | 'executor'
  | 'tool_integration'
  | 'split_strategy'
  | 'text_preprocessor'
  | 'text_postprocessor'
  | 'metadata_provider'
  | 'audio_converter'
  | 'tts_backend'

export interface Extension {
  id?: string
  name: string
  version: string
  type?: string
  extension_type?: ExtensionType | null
  description?: string
  author?: string
  tags?: string[]
  enabled: boolean
  initialized?: boolean
  config?: Record<string, unknown>
  dependencies?: string[]
  installed_at?: string
  source_path?: string
  has_webui?: boolean
  files?: string[]
  path?: string
}

export interface ExtensionWebUIConfig {
  extension_name: string
  route: string
  component_name: string
  mode?: string
  icon?: string
  title: string
  description?: string
}

export interface VoiceInfo {
  name: string
  display_name: string
  language: string
  gender: 'Male' | 'Female'
  locale: string
}

export interface VoiceGroup {
  language: string
  voices: VoiceInfo[]
}

export interface SplitParams {
  input_path: string
  output_dir: string
  preset?: string
  custom_rules?: unknown[]
}

export interface ClusterStatus {
  running: boolean
  node_service_running: boolean
  nodes: { total: number; active: number }
  tasks: { total: number; completed: number; failed: number; pending: number }
}

export interface NodeInfo {
  node_id: string
  host: string
  port: number
  status: 'active' | 'inactive' | 'unhealthy' | 'draining'
  max_concurrency: number
  current_concurrency: number
  total_requests: number
  successful_requests: number
  failed_requests: number
  success_rate: number
  avg_response_time: number
  last_health_check: string | null
  added_at: string
}

export interface ClusterMetricsResponse {
  cluster: {
    total_nodes: number
    active_nodes: number
    total_requests: number
    total_success: number
    total_failure: number
    cluster_avg_latency: number
    cluster_throughput: number
    cluster_success_rate: number
    uptime_seconds: number
  }
  nodes: Record<string, Record<string, unknown>>
}

export interface TaskAssignment {
  task_id: string
  text: string
  voice: string
  rate: string
  output_path: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'retrying'
  assigned_node: string | null
  attempts: number
  error: string | null
  duration: number
  created_at: string
  started_at: string | null
  completed_at: string | null
}

export interface PPCConfig {
  key: string
  value: string
  default_value: string
  type: 'string' | 'number' | 'boolean' | 'select' | 'array' | 'object'
  description?: string
  category: string
  required?: boolean
  editable?: boolean
  options?: string[]
}

export interface ApiErrorDetail {
  message: string
  code?: string
}

export interface ApiResponse<T = unknown> {
  success: boolean
  data?: T
  error?: ApiErrorDetail
}