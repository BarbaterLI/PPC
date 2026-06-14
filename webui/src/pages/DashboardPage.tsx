import { useMemo, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { PageHeader } from '@/components/PageHeader'
import { useApi } from '@/hooks/useApi'
import type { DashboardSystemStatus, TaskInfo } from '@/types'
import { StatusBadge } from '@/components/StatusBadge'
import {
  makeStyles,
  tokens,
  shorthands,
  Text,
  ProgressBar,
  Button,
  Badge,
} from '@fluentui/react-components'
import {
  Play24Regular,
  CheckmarkCircle24Regular,
  Settings24Regular,
  DataTrending24Regular,
  Desktop24Regular,
  DataUsage24Regular,
  Storage24Regular,
  WindowDevTools24Regular,
} from '@fluentui/react-icons'

const MOBILE = '@media (max-width: 767px)'

const useStyles = makeStyles({
  content: {
    ...shorthands.padding(tokens.spacingVerticalXL, tokens.spacingHorizontalXL),
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalXXL),
    maxWidth: '1200px',
    margin: '0 auto',
  },
  sectionTitle: {
    fontWeight: tokens.fontWeightSemibold,
    fontSize: tokens.fontSizeBase500,
    color: tokens.colorNeutralForeground1,
    marginBottom: tokens.spacingVerticalL,
  },
  topSection: {
    display: 'grid',
    gridTemplateColumns: '1fr 1.5fr',
    ...shorthands.gap(tokens.spacingVerticalXL),
    [MOBILE]: {
      gridTemplateColumns: '1fr',
    },
  },
  healthPanel: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    ...shorthands.padding(tokens.spacingVerticalXXL, tokens.spacingHorizontalXL),
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusXLarge,
  },
  healthCircleContainer: {
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  healthCircleSvg: {
    transform: 'rotate(-90deg)',
  },
  healthCircleBg: {
    fill: 'none',
    stroke: tokens.colorNeutralStrokeAlpha,
    strokeWidth: '6',
  },
  healthCircleFill: {
    fill: 'none',
    strokeWidth: '6',
    strokeLinecap: 'round',
    transition: 'stroke-dashoffset 0.8s cubic-bezier(0.34, 1.56, 0.64, 1), stroke 0.6s ease',
  },
  healthScoreText: {
    position: 'absolute',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  healthStatusText: {
    fontSize: tokens.fontSizeBase400,
    fontWeight: tokens.fontWeightSemibold,
    marginTop: tokens.spacingVerticalXS,
  },
  healthSubtext: {
    fontSize: tokens.fontSizeBase300,
    color: tokens.colorNeutralForeground3,
    marginTop: tokens.spacingVerticalM,
  },
  resourceGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gridTemplateRows: '1fr 1fr',
    ...shorthands.gap(tokens.spacingVerticalL, tokens.spacingHorizontalL),
    [MOBILE]: {
      gridTemplateColumns: '1fr',
      gridTemplateRows: 'auto',
    },
  },
  resourceTile: {
    ...shorthands.padding(tokens.spacingVerticalL, tokens.spacingHorizontalL),
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalS),
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusXLarge,
    transition: 'transform 0.2s ease, box-shadow 0.2s ease',
    '&:hover': {
      transform: 'translateY(-2px)',
      boxShadow: tokens.shadow8,
    },
  },
  resourceTileHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    ...shorthands.gap(tokens.spacingHorizontalS),
  },
  resourceIcon: {
    color: tokens.colorBrandForeground1,
    flexShrink: 0,
  },
  resourceValue: {
    fontSize: tokens.fontSizeHero900,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
  },
  resourceLabel: {
    fontSize: tokens.fontSizeBase300,
    color: tokens.colorNeutralForeground3,
  },
  resourceProgress: {
    width: '100%',
    marginTop: tokens.spacingVerticalM,
  },
  processRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    ...shorthands.gap(tokens.spacingHorizontalS),
  },
  processLabel: {
    fontSize: tokens.fontSizeBase300,
    color: tokens.colorNeutralForeground3,
  },
  processValue: {
    fontSize: tokens.fontSizeBase400,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
  },
  tasksSection: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalM),
  },
  taskRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    ...shorthands.padding(tokens.spacingVerticalM, tokens.spacingHorizontalL),
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusLarge,
    ...shorthands.gap(tokens.spacingHorizontalL),
    transition: 'background-color 0.15s ease',
    '&:hover': {
      backgroundColor: tokens.colorNeutralBackground3,
    },
    [MOBILE]: {
      flexDirection: 'column',
      alignItems: 'flex-start',
    },
  },
  taskLeft: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalL),
    minWidth: 0,
    flex: 1,
  },
  taskRight: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalL),
    flexShrink: 0,
  },
  taskLabel: {
    fontSize: tokens.fontSizeBase400,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  taskMeta: {
    fontSize: tokens.fontSizeBase300,
    color: tokens.colorNeutralForeground3,
    whiteSpace: 'nowrap',
  },
  quickActions: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    ...shorthands.gap(tokens.spacingHorizontalL),
    [MOBILE]: {
      gridTemplateColumns: 'repeat(2, 1fr)',
    },
  },
  quickBtn: {
    minHeight: '80px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    ...shorthands.gap(tokens.spacingVerticalS),
    ...shorthands.padding(tokens.spacingVerticalL, tokens.spacingHorizontalL),
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusXLarge,
    transition: 'transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease',
    '&:hover': {
      transform: 'translateY(-3px)',
      boxShadow: tokens.shadow8,
    },
  },
  quickBtnPrimary: {
    backgroundColor: tokens.colorBrandBackground,
    '&:hover': {
      backgroundColor: tokens.colorBrandBackgroundHover,
    },
  },
  quickBtnIcon: {
    fontSize: '32px',
  },
  quickBtnText: {
    fontSize: tokens.fontSizeBase300,
    fontWeight: tokens.fontWeightSemibold,
  },
  loadingContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    ...shorthands.padding(tokens.spacingVerticalXXXL, tokens.spacingHorizontalL),
  },
  errorContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    ...shorthands.padding(tokens.spacingVerticalXXXL, tokens.spacingHorizontalL),
    color: tokens.colorPaletteRedForeground1,
  },
  emptyTasks: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    ...shorthands.padding(tokens.spacingVerticalXXL, tokens.spacingHorizontalL),
    color: tokens.colorNeutralForeground3,
    fontSize: tokens.fontSizeBase400,
  },
})

const CIRCLE_RADIUS = 52
const CIRCLE_CIRCUMFERENCE = 2 * Math.PI * CIRCLE_RADIUS
const CIRCLE_SIZE = (CIRCLE_RADIUS + 8) * 2

function getHealthColor(score: number): string {
  if (score >= 90) return tokens.colorPaletteGreenForeground1
  if (score >= 70) return tokens.colorPaletteYellowForeground1
  return tokens.colorPaletteRedForeground1
}

function getHealthLabel(score: number): string {
  if (score >= 90) return '优秀'
  if (score >= 70) return '良好'
  return '需关注'
}

function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400)
  const h = Math.floor((seconds % 86400) / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (d > 0) return `${d}d ${h}h ${m}m`
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

function formatDateTime(value: string | number): string {
  try {
    const d = typeof value === 'number' ? new Date(value * 1000) : new Date(value)
    return d.toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return String(value)
  }
}

function formatTaskType(type: string): string {
  switch (type) {
    case 'tts': return 'TTS 转换'
    case 'batch': return '批量处理'
    case 'analysis': return '分析'
    case 'convert': return 'TTS 转换'
    case 'analyze': return '分析'
    case 'fanqie_install': return '番茄安装'
    default: return type
  }
}

function taskStatusType(status: string) {
  switch (status) {
    case 'running': return 'running' as const
    case 'completed': return 'completed' as const
    case 'failed': return 'failed' as const
    case 'pending': return 'pending' as const
    case 'cancelled': return 'cancelled' as const
    default: return 'pending' as const
  }
}

export default function DashboardPage() {
  const styles = useStyles()
  const navigate = useNavigate()

  useEffect(() => { document.title = '仪表盘 - PPC10' }, [])

  const { data: status, loading: statusLoading, error: statusError } = useApi<DashboardSystemStatus>(
    '/api/status',
    { refreshInterval: 10000 },
  )

  const { data: tasks } = useApi<TaskInfo[]>('/api/tasks')

  const healthScore = useMemo(
    () => Math.round(status?.health_score ?? 0),
    [status],
  )

  const healthColor = useMemo(() => getHealthColor(healthScore), [healthScore])
  const healthLabel = useMemo(() => getHealthLabel(healthScore), [healthScore])

  const cpuPercent = useMemo(() => status?.cpu_percent ?? 0, [status])
  const memoryPercent = useMemo(() => status?.memory_percent ?? 0, [status])
  const diskPercent = useMemo(() => status?.disk_percent ?? 0, [status])

  const strokeDashoffset = CIRCLE_CIRCUMFERENCE * (1 - healthScore / 100)

  const recentTasks = useMemo(() => {
    if (!tasks) return []
    return [...tasks]
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      .slice(0, 5)
  }, [tasks])

  return (
    <div>
      <PageHeader
        title="仪表盘"
        description="系统运行状态概览"
        actions={
          status && (
            <Badge
              appearance="filled"
              color={status.status === 'running' ? 'success' : status.status === 'error' ? 'danger' : 'warning'}
            >
              {status.status === 'running' ? '运行中' : status.status === 'error' ? '异常' : '未知'}
            </Badge>
          )
        }
      />

      <div className={styles.content}>
        {statusLoading && !status ? (
          <div className={styles.loadingContainer}>
            <Text size={400}>加载中...</Text>
          </div>
        ) : statusError ? (
          <div className={styles.errorContainer}>
            <Text size={400}>无法获取系统状态: {statusError}</Text>
          </div>
        ) : status ? (
          <>
            <div className={styles.topSection}>
              <div className={styles.healthPanel}>
                <div className={styles.healthCircleContainer}>
                  <svg
                    className={styles.healthCircleSvg}
                    width={CIRCLE_SIZE}
                    height={CIRCLE_SIZE}
                    viewBox={`0 0 ${CIRCLE_SIZE} ${CIRCLE_SIZE}`}
                    role="img"
                    aria-label={`健康评分: ${healthScore}`}
                  >
                    <circle
                      className={styles.healthCircleBg}
                      cx={CIRCLE_SIZE / 2}
                      cy={CIRCLE_SIZE / 2}
                      r={CIRCLE_RADIUS}
                    />
                    <circle
                      className={styles.healthCircleFill}
                      cx={CIRCLE_SIZE / 2}
                      cy={CIRCLE_SIZE / 2}
                      r={CIRCLE_RADIUS}
                      stroke={healthColor}
                      strokeDasharray={CIRCLE_CIRCUMFERENCE}
                      strokeDashoffset={strokeDashoffset}
                    />
                  </svg>
                  <div className={styles.healthScoreText}>
                    <Text size={900} weight="semibold" style={{ color: healthColor }}>
                      {healthScore}
                    </Text>
                    <Text className={styles.healthStatusText} style={{ color: healthColor }}>
                      {healthLabel}
                    </Text>
                  </div>
                </div>
                <Text className={styles.healthSubtext}>系统健康评分</Text>
              </div>

              <div className={styles.resourceGrid}>
                <div className={styles.resourceTile}>
                  <div className={styles.resourceTileHeader}>
                    <div className={styles.resourceIcon}>
                      <Desktop24Regular />
                    </div>
                    <Text className={styles.resourceValue}>
                      {cpuPercent.toFixed(1)}%
                    </Text>
                  </div>
                  <Text className={styles.resourceLabel}>CPU 使用率</Text>
                  <ProgressBar
                    className={styles.resourceProgress}
                    value={cpuPercent / 100}
                    color={cpuPercent >= 90 ? 'error' : cpuPercent >= 70 ? 'warning' : 'success'}
                  />
                </div>

                <div className={styles.resourceTile}>
                  <div className={styles.resourceTileHeader}>
                    <div className={styles.resourceIcon}>
                      <DataUsage24Regular />
                    </div>
                    <Text className={styles.resourceValue}>
                      {memoryPercent.toFixed(1)}%
                    </Text>
                  </div>
                  <Text className={styles.resourceLabel}>内存使用率</Text>
                  <ProgressBar
                    className={styles.resourceProgress}
                    value={memoryPercent / 100}
                    color={memoryPercent >= 90 ? 'error' : memoryPercent >= 70 ? 'warning' : 'success'}
                  />
                </div>

                <div className={styles.resourceTile}>
                  <div className={styles.resourceTileHeader}>
                    <div className={styles.resourceIcon}>
                      <Storage24Regular />
                    </div>
                    <Text className={styles.resourceValue}>
                      {diskPercent.toFixed(1)}%
                    </Text>
                  </div>
                  <Text className={styles.resourceLabel}>磁盘使用率</Text>
                  <ProgressBar
                    className={styles.resourceProgress}
                    value={diskPercent / 100}
                    color={diskPercent >= 90 ? 'error' : diskPercent >= 70 ? 'warning' : 'success'}
                  />
                </div>

                <div className={styles.resourceTile}>
                  <div className={styles.resourceTileHeader}>
                    <div className={styles.resourceIcon}>
                      <WindowDevTools24Regular />
                    </div>
                  </div>
                  <div className={styles.processRow}>
                    <Text className={styles.processLabel}>版本</Text>
                    <Text className={styles.processValue}>{status.version}</Text>
                  </div>
                  <div className={styles.processRow}>
                    <Text className={styles.processLabel}>活跃任务</Text>
                    <Text className={styles.processValue}>{status.active_tasks}</Text>
                  </div>
                  <div className={styles.processRow}>
                    <Text className={styles.processLabel}>运行时间</Text>
                    <Text className={styles.processValue}>{formatUptime(status.uptime ?? 0)}</Text>
                  </div>
                </div>
              </div>
            </div>

            <div className={styles.tasksSection}>
              <Text className={styles.sectionTitle}>最近任务</Text>
              {recentTasks.length === 0 ? (
                <div className={styles.emptyTasks}>
                  <Text>暂无任务记录</Text>
                </div>
              ) : (
                recentTasks.map((task) => (
                  <div key={task.task_id} className={styles.taskRow}>
                    <div className={styles.taskLeft}>
                      <StatusBadge status={taskStatusType(task.status)} />
                      <Text className={styles.taskLabel}>{formatTaskType(task.task_type)}</Text>
                    </div>
                    <div className={styles.taskRight}>
                      <Text className={styles.taskMeta}>{formatDateTime(task.created_at)}</Text>
                    </div>
                  </div>
                ))
              )}
            </div>

            <div>
              <Text className={styles.sectionTitle}>快捷操作</Text>
              <div className={styles.quickActions}>
                <Button
                  className={`${styles.quickBtn} ${styles.quickBtnPrimary}`}
                  icon={<Play24Regular className={styles.quickBtnIcon} />}
                  onClick={() => navigate('/tts')}
                >
                  <span className={styles.quickBtnText}>开始转换</span>
                </Button>
                <Button
                  className={styles.quickBtn}
                  icon={<CheckmarkCircle24Regular className={styles.quickBtnIcon} />}
                  onClick={() => navigate('/analysis')}
                >
                  <span className={styles.quickBtnText}>系统检查</span>
                </Button>
                <Button
                  className={styles.quickBtn}
                  icon={<Settings24Regular className={styles.quickBtnIcon} />}
                  onClick={() => navigate('/config')}
                >
                  <span className={styles.quickBtnText}>配置管理</span>
                </Button>
                <Button
                  className={styles.quickBtn}
                  icon={<DataTrending24Regular className={styles.quickBtnIcon} />}
                  onClick={() => navigate('/analysis')}
                >
                  <span className={styles.quickBtnText}>系统分析</span>
                </Button>
              </div>
            </div>
          </>
        ) : null}
      </div>
    </div>
  )
}
