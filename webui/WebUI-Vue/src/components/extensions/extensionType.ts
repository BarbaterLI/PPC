/**
 * 扩展类型标签映射 —— 复刻 React 工程 `src/components/extensions/extensionType.ts`。
 */
import type { Extension } from '@/api/types'

export function extensionTypeLabel(type: string | undefined): string {
  switch (type) {
    case 'tts_backend':
      return 'TTS 后端'
    case 'audio_converter':
      return '音频转换'
    case 'split_strategy':
      return '分割策略'
    case 'text_preprocessor':
      return '文本预处理'
    case 'text_postprocessor':
      return '文本后处理'
    case 'metadata_provider':
      return '元数据提供'
    case 'tool_integration':
      return '工具集成'
    case 'executor':
      return '执行器'
    default:
      return type || '插件'
  }
}

export function getExtensionTypeLabel(ext: Extension): string {
  return ext.extension_type
    ? extensionTypeLabel(ext.extension_type)
    : (ext.type ?? (ext.has_webui ? 'WebUI' : '插件'))
}