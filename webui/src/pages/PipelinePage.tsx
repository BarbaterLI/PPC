import { useState, useEffect, useCallback } from 'react'
import { PageHeader } from '@/components/PageHeader'
import { EmptyState } from '@/components/EmptyState'
import { StatusBadge } from '@/components/StatusBadge'
import { useApi } from '@/hooks/useApi'
import { useToast } from '@/components/ToastNotification'
import type { PipelineInfo, PipelineRun, StepType } from '@/types'
import {
  makeStyles,
  tokens,
  shorthands,
  Text,
  Button,
  Table,
  TableHeader,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  Spinner,
  Tooltip,
  Divider,
  ProgressBar,
} from '@fluentui/react-components'
import {
  Pipeline24Regular,
  Play24Regular,
  ArrowSync24Regular,
} from '@fluentui/react-icons'

const useStyles = makeStyles({
  content: {
    ...shorthands.padding(0, tokens.spacingHorizontalL, tokens.spacingVerticalL),
  },
  tableWrapper: {
    overflowX: 'auto',
  },
  table: {
    width: '100%',
  },
  actionCell: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalXS),
  },
  pipelineName: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalS),
  },
  pipelineIcon: {
    color: tokens.colorBrandForeground1,
    flexShrink: 0,
  },
  loadingContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: tokens.spacingVerticalXXL,
    paddingBottom: tokens.spacingVerticalXXL,
  },
  errorContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: tokens.spacingVerticalXXL,
    paddingBottom: tokens.spacingVerticalXXL,
    ...shorthands.gap(tokens.spacingVerticalM),
  },
  sectionTitle: {
    fontWeight: tokens.fontWeightSemibold,
    fontSize: tokens.fontSizeBase500,
    color: tokens.colorNeutralForeground1,
    marginBottom: tokens.spacingVerticalM,
    marginTop: tokens.spacingVerticalXL,
  },
  runPanel: {
    ...shorthands.padding(tokens.spacingVerticalL, tokens.spacingHorizontalL),
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusXLarge,
    marginTop: tokens.spacingVerticalL,
  },
  runPanelHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: tokens.spacingVerticalM,
  },
  stepRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    ...shorthands.padding(tokens.spacingVerticalS, tokens.spacingHorizontalM),
    ...shorthands.gap(tokens.spacingHorizontalM),
    borderRadius: tokens.borderRadiusMedium,
    marginBottom: tokens.spacingVerticalXS,
    backgroundColor: tokens.colorNeutralBackground1,
  },
  stepLeft: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalM),
    minWidth: 0,
    flex: 1,
  },
  stepName: {
    fontSize: tokens.fontSizeBase300,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  stepMeta: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
    whiteSpace: 'nowrap',
  },
  stepTypeError: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorPaletteRedForeground1,
    maxWidth: '300px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  stepTypeTable: {
    width: '100%',
  },
  monoCell: {
    fontFamily: 'monospace',
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground2,
  },
  divider: {
    marginTop: tokens.spacingVerticalL,
    marginBottom: tokens.spacingVerticalL,
  },
  sourceCell: {
    fontFamily: 'monospace',
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
  },
})

function runStatusToBadge(status: string): 'running' | 'completed' | 'failed' | 'pending' | 'cancelled' | 'warning' {
  switch (status) {
    case 'running': return 'running'
    case 'completed': case 'success': return 'completed'
    case 'failed': case 'error': return 'failed'
    case 'pending': case 'queued': return 'pending'
    case 'cancelled': return 'cancelled'
    case 'warning': return 'warning'
    default: return 'pending'
  }
}

function runStatusLabel(status: string): string {
  switch (status) {
    case 'running': return '运行中'
    case 'completed': case 'success': return '已完成'
    case 'failed': case 'error': return '失败'
    case 'pending': case 'queued': return '等待中'
    case 'cancelled': return '已取消'
    case 'warning': return '警告'
    default: return status
  }
}

function formatDuration(seconds: number): string {
  if (seconds < 1) return '<1s'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

export default function PipelinePage() {
  const styles = useStyles()
  const { showToast } = useToast()

  useEffect(() => { document.title = '管道管理 - PPC10' }, [])

  const { data: pipelines, loading: pipelinesLoading, error: pipelinesError, refetch: refetchPipelines } = useApi<PipelineInfo[]>('/api/pipelines')
  const { data: stepTypes, loading: stepTypesLoading } = useApi<StepType[]>('/api/pipelines/step-types')

  const [currentRun, setCurrentRun] = useState<PipelineRun | null>(null)
  const [running, setRunning] = useState(false)
  const [pollTimer, setPollTimer] = useState<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (pollTimer) {
      clearInterval(pollTimer)
      setPollTimer(null)
    }
  }, [pollTimer])

  const pollRunStatus = useCallback(async (runId: string) => {
    try {
      const res = await fetch(`/api/pipelines/runs/${encodeURIComponent(runId)}`)
      if (!res.ok) throw new Error(`获取运行状态失败: ${res.status}`)
      const data = await res.json()
      const run: PipelineRun = data.data ?? data
      setCurrentRun(run)

      if (run.status !== 'running' && run.status !== 'pending' && run.status !== 'queued') {
        stopPolling()
        setRunning(false)
        if (run.status === 'completed' || run.status === 'success') {
          showToast({ title: '管道执行完成', body: `管道 "${run.pipeline_name}" 执行成功`, intent: 'success' })
        } else if (run.status === 'failed' || run.status === 'error') {
          showToast({ title: '管道执行失败', body: `管道 "${run.pipeline_name}" 执行失败`, intent: 'error' })
        }
      }
    } catch {
      stopPolling()
      setRunning(false)
    }
  }, [stopPolling, showToast])

  const handleRun = useCallback(async (pipeline: PipelineInfo) => {
    setRunning(true)
    setCurrentRun(null)

    try {
      const res = await fetch(`/api/pipelines/${encodeURIComponent(pipeline.name)}/run`, {
        method: 'POST',
      })

      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        throw new Error(errText || `启动管道失败: ${res.status}`)
      }

      const data = await res.json()
      const runData: PipelineRun = data.data ?? data
      setCurrentRun(runData)

      showToast({ title: '管道已启动', body: `管道 "${pipeline.name}" 开始执行`, intent: 'success' })

      const timer = setInterval(() => {
        pollRunStatus(runData.run_id)
      }, 2000)
      setPollTimer(timer)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : '启动管道失败'
      showToast({ title: '启动失败', body: message, intent: 'error' })
      setRunning(false)
    }
  }, [pollRunStatus, showToast])

  useEffect(() => {
    return () => {
      if (pollTimer) clearInterval(pollTimer)
    }
  }, [pollTimer])

  const stepResults = currentRun?.step_results ? Object.values(currentRun.step_results) : []
  const completedSteps = stepResults.filter(s => s.status === 'completed' || s.status === 'success').length
  const totalSteps = stepResults.length

  return (
    <div>
      <PageHeader
        title="管道"
        description="管道流程管理"
        actions={
          <Button
            appearance="subtle"
            icon={<ArrowSync24Regular />}
            onClick={() => { refetchPipelines() }}
            disabled={pipelinesLoading}
          >
            刷新
          </Button>
        }
      />

      <div className={styles.content}>
        {pipelinesLoading && !pipelines ? (
          <div className={styles.loadingContainer}>
            <Spinner label="加载管道列表..." />
          </div>
        ) : pipelinesError ? (
          <div className={styles.errorContainer}>
            <Text>加载失败: {pipelinesError}</Text>
            <Button appearance="secondary" onClick={refetchPipelines}>重试</Button>
          </div>
        ) : !pipelines || pipelines.length === 0 ? (
          <EmptyState
            icon={<Pipeline24Regular fontSize={48} />}
            title="暂无管道"
            message="还没有可用的管道流程。请先配置管道定义文件。"
          />
        ) : (
          <div className={styles.tableWrapper}>
            <Table className={styles.table} size="small">
              <TableHeader>
                <TableRow>
                  <TableHeaderCell>名称</TableHeaderCell>
                  <TableHeaderCell>描述</TableHeaderCell>
                  <TableHeaderCell>步骤数</TableHeaderCell>
                  <TableHeaderCell>来源</TableHeaderCell>
                  <TableHeaderCell>操作</TableHeaderCell>
                </TableRow>
              </TableHeader>
              <TableBody>
                {pipelines.map((p) => (
                  <TableRow key={p.id}>
                    <TableCell>
                      <div className={styles.pipelineName}>
                        <span className={styles.pipelineIcon}>
                          <Pipeline24Regular />
                        </span>
                        <Text weight="semibold">{p.name}</Text>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Text size={200} style={{ color: tokens.colorNeutralForeground3 }}>
                        {p.description || '-'}
                      </Text>
                    </TableCell>
                    <TableCell>
                      <Text size={200}>{p.step_count}</Text>
                    </TableCell>
                    <TableCell>
                      <span className={styles.sourceCell}>{p.source}</span>
                    </TableCell>
                    <TableCell>
                      <div className={styles.actionCell}>
                        <Tooltip content="执行管道" relationship="label">
                          <Button
                            size="small"
                            appearance="subtle"
                            icon={<Play24Regular />}
                            disabled={running}
                            onClick={() => handleRun(p)}
                          >
                            运行
                          </Button>
                        </Tooltip>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}

        {currentRun && (
          <div className={styles.runPanel}>
            <div className={styles.runPanelHeader}>
              <div>
                <Text weight="semibold" size={400}>
                  运行状态: {currentRun.pipeline_name}
                </Text>
                <Text size={200} style={{ display: 'block', color: tokens.colorNeutralForeground3, marginTop: tokens.spacingVerticalXS }}>
                  运行 ID: {currentRun.run_id}
                </Text>
              </div>
              <StatusBadge
                status={runStatusToBadge(currentRun.status)}
                label={runStatusLabel(currentRun.status)}
              />
            </div>

            {totalSteps > 0 && (
              <ProgressBar
                value={completedSteps / totalSteps}
                color={currentRun.status === 'failed' || currentRun.status === 'error' ? 'error' : 'brand'}
              />
            )}

            {totalSteps > 0 && (
              <Text size={200} style={{ color: tokens.colorNeutralForeground3, display: 'block', marginTop: tokens.spacingVerticalXS, marginBottom: tokens.spacingVerticalM }}>
                {completedSteps} / {totalSteps} 步骤完成
                {currentRun.duration_seconds > 0 && ` · 耗时 ${formatDuration(currentRun.duration_seconds)}`}
              </Text>
            )}

            {stepResults.map((step) => (
              <div key={step.step_name} className={styles.stepRow}>
                <div className={styles.stepLeft}>
                  <StatusBadge
                    status={runStatusToBadge(step.status)}
                    label={runStatusLabel(step.status)}
                  />
                  <Text className={styles.stepName}>{step.step_name}</Text>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacingHorizontalM }}>
                  {step.duration_seconds > 0 && (
                    <Text className={styles.stepMeta}>{formatDuration(step.duration_seconds)}</Text>
                  )}
                  {step.error && (
                    <Tooltip content={step.error} relationship="label">
                      <Text className={styles.stepTypeError}>{step.error}</Text>
                    </Tooltip>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        <Divider className={styles.divider} />

        <Text className={styles.sectionTitle}>可用步骤类型</Text>
        {stepTypesLoading ? (
          <div className={styles.loadingContainer}>
            <Spinner size="small" label="加载步骤类型..." />
          </div>
        ) : stepTypes && stepTypes.length > 0 ? (
          <div className={styles.tableWrapper}>
            <Table className={styles.stepTypeTable} size="small">
              <TableHeader>
                <TableRow>
                  <TableHeaderCell>名称</TableHeaderCell>
                  <TableHeaderCell>输入类型</TableHeaderCell>
                  <TableHeaderCell>输出类型</TableHeaderCell>
                </TableRow>
              </TableHeader>
              <TableBody>
                {stepTypes.map((st) => (
                  <TableRow key={st.name}>
                    <TableCell>
                      <Text weight="semibold">{st.name}</Text>
                    </TableCell>
                    <TableCell>
                      <span className={styles.monoCell}>{st.input_type}</span>
                    </TableCell>
                    <TableCell>
                      <span className={styles.monoCell}>{st.output_type}</span>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        ) : (
          <Text size={200} style={{ color: tokens.colorNeutralForeground3 }}>
            暂无可用步骤类型
          </Text>
        )}
      </div>
    </div>
  )
}
