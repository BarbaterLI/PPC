import { useState, useCallback, useEffect } from 'react'
import { PageHeader } from '@/components/PageHeader'
import { useToast } from '@/components/ToastNotification'
import { useSSE } from '@/hooks/useSSE'
import type { AnalyzerStat, AnalysisIssueRaw } from '@/types'
import {
  makeStyles,
  tokens,
  shorthands,
  Text,
  Button,
  Card,
  CardHeader,
} from '@fluentui/react-components'

import { AnalysisModuleSelector } from '@/components/analysis/AnalysisModuleSelector'
import { AnalysisProgress } from '@/components/analysis/AnalysisProgress'
import { AnalysisResultCards } from '@/components/analysis/AnalysisResultCards'
import { AnalysisModuleScores } from '@/components/analysis/AnalysisModuleScores'
import { AnalysisIssuesTable } from '@/components/analysis/AnalysisIssuesTable'
import { AnalysisHistoryTable } from '@/components/analysis/AnalysisHistoryTable'

const useStyles = makeStyles({
  content: {
    ...shorthands.padding(0, tokens.spacingHorizontalL, tokens.spacingVerticalL),
  },
  section: {
    marginBottom: tokens.spacingVerticalXL,
  },
  sectionTitle: {
    fontWeight: tokens.fontWeightSemibold,
    fontSize: tokens.fontSizeBase400,
    color: tokens.colorNeutralForeground1,
    marginBottom: tokens.spacingVerticalM,
  },
  card: {
    ...shorthands.padding(tokens.spacingVerticalM, tokens.spacingHorizontalL),
  },
  cardValue: {
    fontSize: tokens.fontSizeBase600,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
  },
  reportDialog: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalL),
  },
  reportSummary: {
    fontSize: tokens.fontSizeBase300,
    color: tokens.colorNeutralForeground2,
    lineHeight: tokens.lineHeightBase300,
  },
  emptyState: {
    fontSize: tokens.fontSizeBase300,
    color: tokens.colorNeutralForeground2,
    textAlign: 'center',
    ...shorthands.padding(tokens.spacingVerticalXL, 0),
  },
})

interface AnalysisHistoryItem {
  id: string
  task_id?: string
  date: string
  score: number
  analyzers: string[]
  summary?: string
}

interface AnalysisIssueItem {
  analyzer: string
  severity: string
  message: string
  suggestion?: string
}

interface ModuleScore {
  module: string
  score: number
}

interface AnalysisFullReport {
  id: string
  date: string
  overall_score: number
  module_scores: ModuleScore[]
  issues: AnalysisIssueItem[]
  summary?: string
}

interface AnalysisProgress {
  task_id: string
  status: string
  percent: number
  message?: string
}

function getScoreColor(score: number) {
  if (score >= 80) return tokens.colorPaletteGreenForeground1
  if (score >= 60) return tokens.colorPaletteYellowForeground1
  return tokens.colorPaletteRedForeground1
}

export default function AnalysisPage() {
  const styles = useStyles()
  const { showToast } = useToast()

  useEffect(() => { document.title = '系统分析 - PPC10' }, [])

  const [selectedModules, setSelectedModules] = useState<Set<string>>(
    new Set(['performance', 'config', 'errors']),
  )
  const [analyzing, setAnalyzing] = useState(false)
  const [progress, setProgress] = useState<AnalysisProgress | null>(null)
  const [report, setReport] = useState<AnalysisFullReport | null>(null)
  const [historyExpanded, setHistoryExpanded] = useState(false)
  const [selectedReport, setSelectedReport] = useState<AnalysisFullReport | null>(null)
  const [historyData, setHistoryData] = useState<AnalysisHistoryItem[] | null>(null)
  const [historyLoading, setHistoryLoading] = useState(false)

  const { connect: connectSSE, close: closeSSE } = useSSE()

  const toggleModule = useCallback((key: string) => {
    setSelectedModules((prev) => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }, [])

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true)
    try {
      const res = await fetch('/api/analyze/history', {
        headers: { Accept: 'application/json' },
      })
      if (!res.ok) {
        const err = await res.text().catch(() => '')
        throw new Error(err || '加载历史记录失败')
      }
      const data = await res.json()
      const items = Array.isArray(data) ? data : (data.data ?? [])
      setHistoryData(items as AnalysisHistoryItem[])
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '加载失败'
      showToast({ title: '加载历史失败', body: msg, intent: 'error' })
    } finally {
      setHistoryLoading(false)
    }
  }, [showToast])

  const toggleHistory = useCallback(() => {
    setHistoryExpanded((prev) => {
      const next = !prev
      if (next && !historyData) {
        loadHistory()
      }
      return next
    })
  }, [historyData, loadHistory])

  useEffect(() => {
    return () => {
      closeSSE()
    }
  }, [closeSSE])

  const handleStartAnalysis = useCallback(async () => {
    if (selectedModules.size === 0) {
      showToast({ title: '请选择分析模块', intent: 'warning' })
      return
    }

    closeSSE()
    setAnalyzing(true)
    setProgress(null)
    setReport(null)

    const modules = Array.from(selectedModules)

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analyzers: modules }),
      })

      if (!res.ok) {
        const err = await res.text().catch(() => '')
        throw new Error(err || '启动分析失败')
      }

      const result = await res.json()
      const taskId = result.task_id || result.data?.task_id

      if (!taskId) {
        throw new Error('未获取到任务 ID')
      }

      connectSSE(`/api/tasks/${encodeURIComponent(taskId)}/stream`, {
        progress: (event) => {
          try {
            const data = JSON.parse(event.data)
            setProgress({
              task_id: taskId,
              status: 'running',
              percent: data.progress ?? 0,
              message: data.message,
            })
          } catch (error) {
            console.warn('Failed to parse analysis progress event', error)
          }
        },
        complete: (event) => {
          closeSSE()
          setAnalyzing(false)

          try {
            const data = JSON.parse(event.data)
            const fullReport: AnalysisFullReport = {
              id: taskId,
              date: new Date().toISOString(),
              overall_score: data.score ?? 0,
              module_scores: (data.analyzer_stats ?? []).map((s: AnalyzerStat) => ({
                module: s.name ?? s.analyzer ?? 'unknown',
                score: s.score ?? 0,
              })),
              issues: (data.issues ?? []).map((issue: AnalysisIssueRaw) => ({
                analyzer: issue.analyzer ?? 'unknown',
                severity: issue.severity ?? 'info',
                message: issue.message ?? '',
                suggestion: issue.suggestion,
              })),
              summary: data.summary ?? `分析完成，综合评分 ${data.score ?? 0} 分`,
            }
            setReport(fullReport)
          } catch (error) {
            console.warn('Failed to parse analysis complete event', error)
            setProgress(null)
          }

          showToast({ title: '分析完成', body: `已分析 ${modules.length} 个模块`, intent: 'success' })
        },
        error: (event) => {
          try {
            const data = JSON.parse((event as MessageEvent).data)
            showToast({
              title: '分析失败',
              body: data.message || '分析过程中发生错误',
              intent: 'error',
            })
          } catch (error) {
            console.warn('Failed to parse analysis error event', error)
          }
          showToast({ title: '连接中断', body: '与分析服务的连接已断开', intent: 'warning' })
          closeSSE()
          setAnalyzing(false)
        },
      })
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '启动分析失败'
      showToast({ title: '分析失败', body: msg, intent: 'error' })
      setAnalyzing(false)
    }
  }, [selectedModules, showToast, closeSSE, connectSSE])

  const handleStopAnalysis = useCallback(() => {
    closeSSE()
    setAnalyzing(false)
    setProgress(null)
    showToast({ title: '分析已停止', intent: 'info' })
  }, [closeSSE, showToast])

  const handleViewHistoryItem = useCallback(
    async (item: AnalysisHistoryItem) => {
      try {
        const res = await fetch(`/api/analyze/${encodeURIComponent(item.id)}`, {
          headers: { Accept: 'application/json' },
        })
        if (res.ok) {
          const data = await res.json()
          setSelectedReport((data.data || data) as AnalysisFullReport)
        } else {
          showToast({ title: '加载报告失败', body: '无法获取报告详情', intent: 'error' })
          setSelectedReport(null)
        }
      } catch (error) {
        console.warn('Failed to load analysis report', error)
        showToast({ title: '加载报告失败', body: '无法获取报告详情', intent: 'error' })
        setSelectedReport(null)
      }
    },
    [],
  )

  return (
    <div>
      <PageHeader
        title="分析"
        description="系统诊断与健康分析"
      />

      <div className={styles.content}>
        <div className={styles.section}>
          <div className={styles.sectionTitle}>分析模块选择</div>
          <AnalysisModuleSelector
            selectedModules={selectedModules}
            onToggleModule={toggleModule}
            onStartAnalysis={handleStartAnalysis}
            onStopAnalysis={handleStopAnalysis}
            onToggleHistory={toggleHistory}
            analyzing={analyzing}
          />
        </div>

        {analyzing && progress && (
          <AnalysisProgress
            progress={progress.percent}
            message={progress.message}
          />
        )}

        {report && !analyzing && (
          <div className={styles.section}>
            <div className={styles.sectionTitle}>分析结果</div>

            <AnalysisResultCards
              overallScore={report.overall_score}
              moduleCount={report.module_scores.length}
              issueCount={report.issues.length}
            />

            <div className={styles.sectionTitle}>模块评分</div>
            <div style={{ marginBottom: tokens.spacingVerticalL }}>
              <AnalysisModuleScores moduleScores={report.module_scores} />
            </div>

            {report.issues.length > 0 && (
              <>
                <div className={styles.sectionTitle}>问题列表</div>
                <AnalysisIssuesTable issues={report.issues} />
              </>
            )}
          </div>
        )}

        {!analyzing && !report && (
          <div className={styles.emptyState}>
            <Text>选择分析模块后点击"开始分析"以运行系统诊断</Text>
          </div>
        )}

        {historyExpanded && (
          <div className={styles.section}>
            <div className={styles.sectionTitle}>历史记录</div>
            <AnalysisHistoryTable
              historyData={historyData}
              historyLoading={historyLoading}
              onViewHistoryItem={handleViewHistoryItem}
            />
          </div>
        )}

        {selectedReport && (
          <div className={styles.section}>
            <div className={styles.sectionTitle}>
              历史报告 — {new Date(selectedReport.date).toLocaleString('zh-CN')}
            </div>
            <div className={styles.reportDialog}>
              <Card className={styles.card}>
                <CardHeader header={<Text weight="semibold">综合评分</Text>} />
                <div
                  className={styles.cardValue}
                  style={{ color: getScoreColor(selectedReport.overall_score) }}
                >
                  {selectedReport.overall_score} / 100
                </div>
              </Card>
              {selectedReport.summary && (
                <div className={styles.reportSummary}>{selectedReport.summary}</div>
              )}
              {selectedReport.module_scores && selectedReport.module_scores.length > 0 && (
                <>
                  <Text weight="semibold">模块评分</Text>
                  <AnalysisModuleScores moduleScores={selectedReport.module_scores} />
                </>
              )}
              {selectedReport.issues && selectedReport.issues.length > 0 && (
                <>
                  <Text weight="semibold">问题列表</Text>
                  <AnalysisIssuesTable issues={selectedReport.issues} />
                </>
              )}
              <Button appearance="secondary" onClick={() => setSelectedReport(null)}>
                关闭报告
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
