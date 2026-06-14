export type AnalysisCategory =
  | 'performance'
  | 'memory'
  | 'reliability'
  | 'configuration'
  | 'security'
  | 'dependency'
  | 'code_quality'
  | 'resource'
  | 'network'
  | 'unknown'

export type RiskLevel = 'critical' | 'high' | 'medium' | 'low' | 'none'

export type Severity = 'critical' | 'high' | 'medium' | 'low' | 'info'

export interface AnalysisIssue {
  severity: Severity
  category: AnalysisCategory
  description: string
  suggestion?: string
  location?: string
  details: Record<string, unknown>
  timestamp: string
}

export interface RepairSuggestion {
  action: string
  risk_level: RiskLevel
  expected_impact: string
  strategy_name?: string
  parameters: Record<string, unknown>
  auto_applicable: boolean
}

export interface HealthReport {
  score: number
  issues: AnalysisIssue[]
  timestamp: string
  component?: string
  summary?: string
  metrics: Record<string, unknown>
}

export interface RepairResult {
  success: boolean
  message: string
  backup_path?: string
  rolled_back: boolean
  error?: string
  metrics: Record<string, unknown>
}

export interface AnalysisResult {
  health_report: HealthReport
  repair_suggestions: RepairSuggestion[]
  analysis_time: number
  modules_run: string[]
}

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

export type LogLevel = 'debug' | 'info' | 'warning' | 'error'
export type UIMode = 'simple' | 'classic' | 'debug'
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

export type TaskStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface TaskProgress {
  task_id: string
  status: TaskStatus
  percent: number
  current_file?: string
  processed: number
  total: number
  speed?: number
  eta?: number
  success_count: number
  fail_count: number
  elapsed: number
  message?: string
}

export interface Task {
  id: string
  task_id: string
  type: string
  task_type: string
  status: TaskStatus
  created_at: string
  started_at?: string
  completed_at?: string
  params: Record<string, unknown>
  result?: Record<string, unknown>
  progress: TaskProgress
  progress_message?: string
}

export interface ConvertTaskParams {
  input_dir: string
  output_dir: string
  voice: string
  concurrency?: number
  format?: AudioFormat
  quality?: AudioQuality
  rate?: string
}

export type ExtensionType =
  | 'load_balance_strategy'
  | 'health_check_strategy'
  | 'task_scheduling_strategy'
  | 'metrics_exporter'
  | 'executor'
  | 'tool_integration'

export interface Extension {
  id: string
  name: string
  version: string
  description: string
  author: string
  extension_type?: ExtensionType
  tags: string[]
  enabled: boolean
  initialized: boolean
  config: Record<string, unknown>
  dependencies: string[]
  installed_at?: string
  source_path?: string
  has_webui: boolean
  files?: string[]
}

export interface ExtensionWebUIConfig {
  extension_name: string
  route: string
  component_name: string
  icon?: string
  title: string
  description?: string
}

export interface FanqieStatus {
  installed: boolean
  version?: string
  latest_version?: string
  update_available: boolean
  exe_path?: string
  server_running: boolean
  server_host?: string
  server_port?: number
}

export interface FanqieConfig {
  use_mirror: boolean
  mirror_host: string
  prefer_musl: boolean
  server_host: string
  server_port: number
}

export interface FanqieInstallParams {
  use_mirror?: boolean
  mirror?: string
  prefer_musl?: boolean
}

export interface FanqieLaunchServerParams {
  host?: string
  port?: number
  password?: string
  data_dir?: string
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

export interface MergeParams {
  input_dir: string
  output_path: string
  format?: AudioFormat
  silence_between_ms?: number
}

export interface PreviewParams {
  text: string
  voice: string
  rate?: string
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

export interface AnalyzerStat {
  name?: string
  analyzer?: string
  score?: number
  issues_count?: number
  [key: string]: unknown
}

export interface AnalysisIssueRaw {
  analyzer?: string
  severity?: string
  message?: string
  suggestion?: string
  location?: string
  [key: string]: unknown
}

export type TaskInfo = Task

export type ExtensionInfo = Extension

export interface PPC10Config {
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

export interface VoiceOption {
  id: string
  name: string
  locale: string
  status: 'available' | 'unavailable'
}

export type DashboardSystemStatus = SystemStatus

export interface ApiResponse<T = unknown> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
}

export type {
  PipelineStep,
  PipelineDefinition,
  PipelineInfo,
  StepResult,
  PipelineRun,
  StepType,
} from './pipeline'
