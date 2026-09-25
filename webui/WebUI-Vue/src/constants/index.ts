/**
 * 业务常量 —— 镜像 React 工程 `src/constants/index.ts` 中与本工程相关部分。
 */

export const DEFAULT_NODE_HOST = '0.0.0.0'

export const DEFAULT_NODE_PORT = 8080

export const DEFAULT_NODE_MAX_CONCURRENCY = 4

export const STORAGE_KEYS = {
  CONFIG_SELECTED_TAB: 'config-selected-tab',
} as const

export const CONFIG_CATEGORY_LABELS: Record<string, string> = {
  core: '核心',
  tts: 'TTS',
  split: '分割',
  batch: '批量',
  performance: '性能',
  network: '网络',
  reliability: '可靠性',
  distributed: '分布式',
  extension: '扩展',
  fanqie: '番茄小说',
  output: '输出',
}

export const CONFIG_CATEGORY_ORDER = [
  'core',
  'tts',
  'split',
  'performance',
  'network',
  'reliability',
  'extension',
  'output',
] as const