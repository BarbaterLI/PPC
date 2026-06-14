import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { PageHeader } from '@/components/PageHeader'
import { useApi } from '@/hooks/useApi'
import { useSSE } from '@/hooks/useSSE'
import { systemApi, taskApi } from '@/api/client'
import type { VoiceGroup } from '@/types'
import { makeStyles, tokens, shorthands } from '@fluentui/react-components'

import { TtsConvertConfigPanel } from '@/components/tts/TtsConvertConfigPanel'
import { TtsConvertProgress } from '@/components/tts/TtsConvertProgress'
import { TtsConvertResult } from '@/components/tts/TtsConvertResult'

const useStyles = makeStyles({
  content: {
    ...shorthands.padding(tokens.spacingVerticalL, tokens.spacingHorizontalL),
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalL),
  },
})

interface SseProgressData {
  progress: number
  current_file?: string
  speed?: string
  eta?: string
  total_files?: number
  completed_files?: number
}

interface ConvertResult {
  success_count: number
  failure_count: number
  duration: number
  failed_items: { file: string; error: string }[]
}

type PageState = 'config' | 'running' | 'completed'

interface ConvertFormData {
  input_dir: string
  output_dir: string
  voice_id: string
  concurrency: number
  rate: string
  recursive: boolean
  resume: boolean
}

const INITIAL_FORM: ConvertFormData = {
  input_dir: '',
  output_dir: '',
  voice_id: '',
  concurrency: 12,
  rate: '+0%',
  recursive: false,
  resume: false,
}

export default function TtsConvertPage() {
  const styles = useStyles()

  useEffect(() => { document.title = 'TTS 转换 - PPC10' }, [])

  const [pageState, setPageState] = useState<PageState>('config')
  const [formData, setFormData] = useState<ConvertFormData>(INITIAL_FORM)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  const [taskId, setTaskId] = useState<string | null>(null)
  const [progressData, setProgressData] = useState<SseProgressData>({ progress: 0 })
  const [convertResult, setConvertResult] = useState<ConvertResult | null>(null)

  const startTimeRef = useRef<number>(0)
  const sseTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const { data: voiceGroups, loading: voicesLoading } = useApi<VoiceGroup[]>(() => systemApi.getVoices())

  const { connect: connectSSE, close: closeSse } = useSSE()

  const voiceOptions = useMemo(() => {
    if (!voiceGroups || !Array.isArray(voiceGroups)) return []
    const options: { value: string; label: string }[] = []
    for (const group of voiceGroups) {
      if (!group.voices || !Array.isArray(group.voices)) continue
      for (const voice of group.voices) {
        options.push({
          value: voice.name,
          label: `${voice.display_name} (${voice.locale})`,
        })
      }
    }
    return options
  }, [voiceGroups])

  const resetSseTimeout = useCallback(() => {
    if (sseTimeoutRef.current) {
      clearTimeout(sseTimeoutRef.current)
    }
    sseTimeoutRef.current = setTimeout(() => {
      closeSse()
      sseTimeoutRef.current = null
      setPageState('config')
      setTaskId(null)
    }, 300000)
  }, [closeSse])

  useEffect(() => {
    return () => {
      closeSse()
    }
  }, [closeSse])

  const handleStartConvert = useCallback(async () => {
    if (!formData.input_dir || !formData.output_dir || !formData.voice_id) {
      setSubmitError('请填写输入目录、输出目录并选择语音')
      return
    }

    setSubmitting(true)
    setSubmitError(null)

    try {
      const result = await taskApi.createConvert({
        input_dir: formData.input_dir,
        output_dir: formData.output_dir,
        voice: formData.voice_id,
        concurrency: formData.concurrency,
        rate: formData.rate,
      })
      if (result.success === false) {
        throw new Error(result.error || result.message || '创建任务失败')
      }
      const id = result.data?.task_id
      if (!id) {
        throw new Error('未获取到任务ID')
      }

      setTaskId(id)
      setPageState('running')
      setProgressData({ progress: 0 })
      startTimeRef.current = Date.now()

      resetSseTimeout()

      connectSSE(`/api/tasks/${id}/stream`, {
        progress: (event) => {
          resetSseTimeout()
          try {
            const data = JSON.parse(event.data) as SseProgressData
            setProgressData(data)
          } catch (error) {
            console.warn('Failed to parse progress event', error)
          }
        },
        complete: (event) => {
          if (sseTimeoutRef.current) {
            clearTimeout(sseTimeoutRef.current)
            sseTimeoutRef.current = null
          }
          try {
            const data = JSON.parse(event.data) as ConvertResult
            setConvertResult(data)
          } catch {
            setConvertResult({
              success_count: 0,
              failure_count: 0,
              duration: (Date.now() - startTimeRef.current) / 1000,
              failed_items: [],
            })
          }
          setPageState('completed')
          closeSse()
        },
        error: () => {
          if (sseTimeoutRef.current) {
            clearTimeout(sseTimeoutRef.current)
            sseTimeoutRef.current = null
          }
          setConvertResult((prev) =>
            prev ?? {
              success_count: 0,
              failure_count: 0,
              duration: (Date.now() - startTimeRef.current) / 1000,
              failed_items: [],
            },
          )
          setPageState('completed')
          closeSse()
        },
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : '启动转换任务失败'
      setSubmitError(message)
    } finally {
      setSubmitting(false)
    }
  }, [formData, connectSSE, closeSse, resetSseTimeout])

  const handleCancel = useCallback(async () => {
    if (!taskId) return

    closeSse()

    try {
      await taskApi.cancel(taskId)
    } catch (error) {
      console.warn('Failed to cancel task', error)
    }

    setPageState('config')
    setTaskId(null)
  }, [taskId, closeSse])

  const handleReset = useCallback(() => {
    setPageState('config')
    setTaskId(null)
    setProgressData({ progress: 0 })
    setConvertResult(null)
  }, [])

  return (
    <div>
      <PageHeader
        title="TTS 转换"
        description="文本转语音批量转换工具"
      />

      <div className={styles.content}>
        {pageState === 'config' && (
          <TtsConvertConfigPanel
            formData={formData}
            setFormData={setFormData}
            voiceOptions={voiceOptions}
            voicesLoading={voicesLoading}
            submitting={submitting}
            submitError={submitError}
            onSubmit={handleStartConvert}
          />
        )}

        {pageState === 'running' && (
          <TtsConvertProgress
            progressData={progressData}
            onCancel={handleCancel}
          />
        )}

        {pageState === 'completed' && convertResult && (
          <TtsConvertResult
            result={convertResult}
            onReset={handleReset}
          />
        )}
      </div>
    </div>
  )
}
