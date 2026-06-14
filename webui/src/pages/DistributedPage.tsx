import { useState, useCallback, useEffect } from 'react'
import { useApi } from '@/hooks/useApi'
import type { ClusterStatus, NodeInfo, ClusterMetricsResponse, TaskAssignment } from '@/types'
import {
  makeStyles,
  shorthands,
  Text,
  Button,
  Badge,
  ProgressBar,
  Spinner,
  Dialog,
  DialogTrigger,
  DialogSurface,
  DialogBody,
  DialogTitle,
  DialogContent,
  DialogActions,
  Input,
  Label,
  Table,
  TableHeader,
  TableRow,
  TableCell,
  TableHeaderCell,
  TableBody,
  MessageBar,
  MessageBarBody,
  MessageBarTitle,
  tokens,
} from '@fluentui/react-components'
import {
  Server24Regular,
  CheckmarkCircle24Regular,
  ArrowTrending24Regular,
  Clock24Regular,
  Add24Regular,
  Play24Regular,
  Stop24Regular,
  Delete24Regular,
  ArrowDownload24Regular,
  Checkmark24Regular,
} from '@fluentui/react-icons'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalL),
  },
  overviewGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    ...shorthands.gap(tokens.spacingHorizontalM),
    '@media (max-width: 900px)': {
      gridTemplateColumns: 'repeat(2, 1fr)',
    },
    '@media (max-width: 500px)': {
      gridTemplateColumns: '1fr',
    },
  },
  overviewCard: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalS),
    ...shorthands.padding(tokens.spacingVerticalL, tokens.spacingHorizontalL),
    backgroundColor: tokens.colorNeutralBackground1,
    ...shorthands.border('1px', 'solid', tokens.colorNeutralStroke1),
    borderRadius: tokens.borderRadiusLarge,
  },
  overviewCardHeader: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalS),
  },
  overviewCardIcon: {
    color: tokens.colorBrandForeground1,
  },
  overviewCardLabel: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
  },
  overviewCardValue: {
    fontSize: tokens.fontSizeBase500,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
  },
  actionBar: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalS),
    flexWrap: 'wrap',
  },
  sectionTitle: {
    fontSize: tokens.fontSizeBase400,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
    marginBottom: tokens.spacingVerticalS,
  },
  tableContainer: {
    overflowX: 'auto',
    backgroundColor: tokens.colorNeutralBackground1,
    ...shorthands.border('1px', 'solid', tokens.colorNeutralStroke1),
    borderRadius: tokens.borderRadiusLarge,
  },
  bottomGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    ...shorthands.gap(tokens.spacingHorizontalM),
    '@media (max-width: 768px)': {
      gridTemplateColumns: '1fr',
    },
  },
  metricCard: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalS),
    ...shorthands.padding(tokens.spacingVerticalL, tokens.spacingHorizontalL),
    backgroundColor: tokens.colorNeutralBackground1,
    ...shorthands.border('1px', 'solid', tokens.colorNeutralStroke1),
    borderRadius: tokens.borderRadiusLarge,
  },
  taskStatsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    ...shorthands.gap(tokens.spacingVerticalS, tokens.spacingHorizontalM),
  },
  taskStatItem: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalXS),
  },
  taskStatLabel: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
  },
  taskStatValue: {
    fontSize: tokens.fontSizeBase400,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
  },
  progressSection: {
    marginTop: tokens.spacingVerticalM,
  },
  progressLabel: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: tokens.spacingVerticalXS,
  },
  metricList: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalM),
  },
  metricRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  metricLabel: {
    fontSize: tokens.fontSizeBase300,
    color: tokens.colorNeutralForeground3,
  },
  metricValue: {
    fontSize: tokens.fontSizeBase300,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
  },
  actionCell: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalXS),
  },
  dialogField: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalXS),
  },
  loadingContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: tokens.spacingVerticalXXL,
    paddingBottom: tokens.spacingVerticalXXL,
  },
  statusBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalXS),
  },
  statusDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
  },
  statusDotActive: {
    backgroundColor: tokens.colorPaletteGreenBackground3,
  },
  statusDotInactive: {
    backgroundColor: tokens.colorNeutralForeground3,
  },
})

function getStatusBadgeAppearance(status: NodeInfo['status']): 'filled' | 'ghost' | 'tint' | 'outline' {
  switch (status) {
    case 'active': return 'filled'
    case 'inactive': return 'ghost'
    case 'unhealthy': return 'outline'
    case 'draining': return 'tint'
  }
}

function getStatusBadgeLabel(status: NodeInfo['status']): string {
  switch (status) {
    case 'active': return '活跃'
    case 'inactive': return '离线'
    case 'unhealthy': return '异常'
    case 'draining': return '排空中'
  }
}

export default function DistributedPage() {
  const styles = useStyles()

  useEffect(() => { document.title = '分布式管理 - PPC10' }, [])

  const { data: clusterStatus, loading: statusLoading, refetch: refetchStatus } = useApi<ClusterStatus>('/api/distributed/status', { refreshInterval: 5000 })
  const { data: nodes, loading: nodesLoading, refetch: refetchNodes } = useApi<NodeInfo[]>('/api/distributed/nodes', { refreshInterval: 5000 })
  const { data: metrics, refetch: refetchMetrics } = useApi<ClusterMetricsResponse>('/api/distributed/metrics', { refreshInterval: 5000 })
  const { refetch: refetchTasks } = useApi<TaskAssignment[]>('/api/distributed/tasks', { refreshInterval: 5000 })

  const [addDialogOpen, setAddDialogOpen] = useState(false)
  const [newHost, setNewHost] = useState('')
  const [newPort, setNewPort] = useState('8080')
  const [newMaxConcurrency, setNewMaxConcurrency] = useState('4')
  const [operating, setOperating] = useState<string | null>(null)
  const [lastError, setLastError] = useState<string | null>(null)

  const schedulerRunning = clusterStatus?.running ?? false
  const nodeServiceRunning = clusterStatus?.node_service_running ?? false

  const refetchAll = useCallback(() => {
    refetchStatus()
    refetchNodes()
    refetchMetrics()
    refetchTasks()
  }, [refetchStatus, refetchNodes, refetchMetrics, refetchTasks])

  const handleApiCall = useCallback(async (label: string, url: string, options?: RequestInit) => {
    setOperating(label)
    setLastError(null)
    try {
      const response = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
      })
      if (!response.ok) {
        let errMsg = `操作失败: ${response.status}`
        try {
          const body = await response.json()
          errMsg = body.error || errMsg
        } catch (error) {
          console.warn('Failed to parse error response', error)
        }
        setLastError(errMsg)
      }
      refetchAll()
    } catch (err) {
      setLastError(err instanceof Error ? err.message : '网络错误')
    } finally {
      setOperating(null)
    }
  }, [refetchAll])

  const handleStartScheduler = useCallback(() => {
    handleApiCall('scheduler-start', '/api/distributed/start', {
      method: 'POST',
      body: JSON.stringify({ strategy: 'round_robin', local_execution: true }),
    })
  }, [handleApiCall])

  const handleStopScheduler = useCallback(() => {
    handleApiCall('scheduler-stop', '/api/distributed/stop', { method: 'POST' })
  }, [handleApiCall])

  const handleStartNodeService = useCallback(() => {
    handleApiCall('node-start', '/api/distributed/node-service/start', {
      method: 'POST',
      body: JSON.stringify({ host: '0.0.0.0', port: 8080, max_concurrency: 4 }),
    })
  }, [handleApiCall])

  const handleStopNodeService = useCallback(() => {
    handleApiCall('node-stop', '/api/distributed/node-service/stop', { method: 'POST' })
  }, [handleApiCall])

  const handleAddNode = useCallback(async () => {
    const port = parseInt(newPort, 10)
    const maxConcurrency = parseInt(newMaxConcurrency, 10)
    if (!newHost || isNaN(port) || isNaN(maxConcurrency)) return

    setOperating('add-node')
    setLastError(null)
    try {
      const response = await fetch('/api/distributed/nodes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ host: newHost, port, max_concurrency: maxConcurrency }),
      })
      if (!response.ok) {
        let errMsg = `添加节点失败: ${response.status}`
        try {
          const body = await response.json()
          errMsg = body.error || errMsg
        } catch (error) {
          console.warn('Failed to parse error response', error)
        }
        setLastError(errMsg)
      } else {
        setAddDialogOpen(false)
        setNewHost('')
        setNewPort('8080')
        setNewMaxConcurrency('4')
      }
      refetchAll()
    } catch (err) {
      setLastError(err instanceof Error ? err.message : '网络错误')
    } finally {
      setOperating(null)
    }
  }, [newHost, newPort, newMaxConcurrency, refetchAll])

  const handleRemoveNode = useCallback(async (nodeId: string) => {
    if (!window.confirm(`确定要移除节点 ${nodeId} 吗？`)) return
    setLastError(null)
    try {
      const response = await fetch(`/api/distributed/nodes/${encodeURIComponent(nodeId)}`, { method: 'DELETE' })
      if (!response.ok) {
        let errMsg = `移除节点失败: ${response.status}`
        try {
          const body = await response.json()
          errMsg = body.error || errMsg
        } catch (error) {
          console.warn('Failed to parse error response', error)
        }
        setLastError(errMsg)
        return
      }
      refetchAll()
    } catch (err) {
      setLastError(err instanceof Error ? err.message : '网络错误')
    }
  }, [refetchAll])

  const handleDrainNode = useCallback(async (nodeId: string) => {
    if (!window.confirm(`确定要排空节点 ${nodeId} 吗？`)) return
    setLastError(null)
    try {
      const response = await fetch(`/api/distributed/nodes/${encodeURIComponent(nodeId)}/drain`, { method: 'POST' })
      if (!response.ok) {
        let errMsg = `排空节点失败: ${response.status}`
        try {
          const body = await response.json()
          errMsg = body.error || errMsg
        } catch { /* ignore */ }
        setLastError(errMsg)
        return
      }
      refetchAll()
    } catch (err) {
      setLastError(err instanceof Error ? err.message : '网络错误')
    }
  }, [refetchAll])

  const handleActivateNode = useCallback(async (nodeId: string) => {
    setLastError(null)
    try {
      const response = await fetch(`/api/distributed/nodes/${encodeURIComponent(nodeId)}/activate`, { method: 'POST' })
      if (!response.ok) {
        let errMsg = `激活节点失败: ${response.status}`
        try {
          const body = await response.json()
          errMsg = body.error || errMsg
        } catch (error) {
          console.warn('Failed to parse error response', error)
        }
        setLastError(errMsg)
        return
      }
      refetchAll()
    } catch (err) {
      setLastError(err instanceof Error ? err.message : '网络错误')
    }
  }, [refetchAll])

  const taskStats = clusterStatus?.tasks ?? { total: 0, completed: 0, failed: 0, pending: 0 }
  const completionRate = taskStats.total > 0 ? taskStats.completed / taskStats.total : 0
  const clusterMetricsData = metrics?.cluster

  return (
    <div className={styles.root}>
      <Text size={500} weight="semibold">分布式集群管理</Text>

      {(statusLoading || nodesLoading) && !clusterStatus && !nodes ? (
        <div className={styles.loadingContainer}>
          <Spinner label="加载集群状态..." />
        </div>
      ) : (
        <>
      {lastError && (
        <MessageBar intent="error">
          <MessageBarBody>
            <MessageBarTitle>错误</MessageBarTitle>
            {lastError}
          </MessageBarBody>
        </MessageBar>
      )}

      <div className={styles.overviewGrid}>
        <div className={styles.overviewCard}>
          <div className={styles.overviewCardHeader}>
            <Server24Regular className={styles.overviewCardIcon} />
            <Text className={styles.overviewCardLabel}>节点总数</Text>
          </div>
          <Text className={styles.overviewCardValue}>{clusterStatus?.nodes?.total ?? 0}</Text>
        </div>
        <div className={styles.overviewCard}>
          <div className={styles.overviewCardHeader}>
            <CheckmarkCircle24Regular className={styles.overviewCardIcon} />
            <Text className={styles.overviewCardLabel}>活跃节点</Text>
          </div>
          <Text className={styles.overviewCardValue}>{clusterStatus?.nodes?.active ?? 0}</Text>
        </div>
        <div className={styles.overviewCard}>
          <div className={styles.overviewCardHeader}>
            <ArrowTrending24Regular className={styles.overviewCardIcon} />
            <Text className={styles.overviewCardLabel}>集群吞吐量</Text>
          </div>
          <Text className={styles.overviewCardValue}>
            {clusterMetricsData?.cluster_throughput != null && clusterMetricsData.cluster_throughput > 0
              ? `${clusterMetricsData.cluster_throughput.toFixed(1)}/s`
              : '0.0/s'}
          </Text>
        </div>
        <div className={styles.overviewCard}>
          <div className={styles.overviewCardHeader}>
            <Clock24Regular className={styles.overviewCardIcon} />
            <Text className={styles.overviewCardLabel}>平均延迟</Text>
          </div>
          <Text className={styles.overviewCardValue}>
            {clusterMetricsData?.cluster_avg_latency != null && clusterMetricsData.cluster_avg_latency > 0
              ? `${clusterMetricsData.cluster_avg_latency.toFixed(0)}ms`
              : '-'}
          </Text>
        </div>
      </div>

      <div className={styles.actionBar}>
        {schedulerRunning ? (
          <Button
            appearance="primary"
            icon={<Stop24Regular />}
            disabled={operating === 'scheduler-stop'}
            onClick={handleStopScheduler}
          >
            停止调度器
          </Button>
        ) : (
          <Button
            appearance="primary"
            icon={<Play24Regular />}
            disabled={operating === 'scheduler-start'}
            onClick={handleStartScheduler}
          >
            启动调度器
          </Button>
        )}
        {nodeServiceRunning ? (
          <Button
            appearance="secondary"
            icon={<Stop24Regular />}
            disabled={operating === 'node-stop'}
            onClick={handleStopNodeService}
          >
            停止本机节点
          </Button>
        ) : (
          <Button
            appearance="secondary"
            icon={<Play24Regular />}
            disabled={operating === 'node-start'}
            onClick={handleStartNodeService}
          >
            启动本机节点
          </Button>
        )}
        <Dialog open={addDialogOpen} onOpenChange={(_, data) => setAddDialogOpen(data.open)}>
          <DialogTrigger disableButtonEnhancement>
            <Button appearance="secondary" icon={<Add24Regular />} disabled={!schedulerRunning}>
              添加节点
            </Button>
          </DialogTrigger>
          <DialogSurface>
            <DialogBody>
              <DialogTitle>添加节点</DialogTitle>
              <DialogContent>
                <div style={{ display: 'flex', flexDirection: 'column', gap: tokens.spacingVerticalM }}>
                  <div className={styles.dialogField}>
                    <Label htmlFor="add-host">主机地址</Label>
                    <Input
                      id="add-host"
                      value={newHost}
                      onChange={(_, data) => setNewHost(data.value)}
                      placeholder="例如: 192.168.1.100"
                    />
                  </div>
                  <div className={styles.dialogField}>
                    <Label htmlFor="add-port">端口</Label>
                    <Input
                      id="add-port"
                      type="number"
                      value={newPort}
                      onChange={(_, data) => setNewPort(data.value)}
                      placeholder="8080"
                    />
                  </div>
                  <div className={styles.dialogField}>
                    <Label htmlFor="add-concurrency">最大并发数</Label>
                    <Input
                      id="add-concurrency"
                      type="number"
                      value={newMaxConcurrency}
                      onChange={(_, data) => setNewMaxConcurrency(data.value)}
                      placeholder="4"
                    />
                  </div>
                </div>
              </DialogContent>
              <DialogActions>
                <Button appearance="primary" disabled={operating === 'add-node'} onClick={handleAddNode}>
                  添加
                </Button>
                <Button appearance="secondary" onClick={() => setAddDialogOpen(false)}>
                  取消
                </Button>
              </DialogActions>
            </DialogBody>
          </DialogSurface>
        </Dialog>
      </div>

      <div>
        <Text className={styles.sectionTitle}>节点管理</Text>
        <div className={styles.tableContainer}>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHeaderCell>节点 ID</TableHeaderCell>
                <TableHeaderCell>地址</TableHeaderCell>
                <TableHeaderCell>状态</TableHeaderCell>
                <TableHeaderCell>并发</TableHeaderCell>
                <TableHeaderCell>成功率</TableHeaderCell>
                <TableHeaderCell>平均延迟</TableHeaderCell>
                <TableHeaderCell>操作</TableHeaderCell>
              </TableRow>
            </TableHeader>
            <TableBody>
              {(!nodes || nodes.length === 0) ? (
                <TableRow>
                  <TableCell colSpan={7}>
                    <div className={styles.loadingContainer}>
                      <Text size={300} style={{ color: tokens.colorNeutralForeground3 }}>
                        {schedulerRunning ? '暂无节点数据，点击"添加节点"添加远程节点' : '请先启动调度器'}
                      </Text>
                    </div>
                  </TableCell>
                </TableRow>
              ) : (
                nodes.map((node) => (
                  <TableRow key={node.node_id}>
                    <TableCell>
                      <Text size={300} style={{ fontFamily: 'monospace' }}>{node.node_id}</Text>
                    </TableCell>
                    <TableCell>
                      <Text size={300} style={{ fontFamily: 'monospace' }}>{node.host}:{node.port}</Text>
                    </TableCell>
                    <TableCell>
                      <Badge appearance={getStatusBadgeAppearance(node.status)} size="small">
                        {getStatusBadgeLabel(node.status)}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Text size={300}>{node.current_concurrency}/{node.max_concurrency}</Text>
                    </TableCell>
                    <TableCell>
                      <Text size={300}>{node.success_rate.toFixed(1)}%</Text>
                    </TableCell>
                    <TableCell>
                      <Text size={300}>{node.avg_response_time.toFixed(0)}ms</Text>
                    </TableCell>
                    <TableCell>
                      <div className={styles.actionCell}>
                        {node.status === 'active' && (
                          <Button
                            appearance="subtle"
                            size="small"
                            icon={<ArrowDownload24Regular />}
                            onClick={() => handleDrainNode(node.node_id)}
                          >
                            排空
                          </Button>
                        )}
                        {(node.status === 'inactive' || node.status === 'draining') && (
                          <Button
                            appearance="subtle"
                            size="small"
                            icon={<Checkmark24Regular />}
                            onClick={() => handleActivateNode(node.node_id)}
                          >
                            激活
                          </Button>
                        )}
                        <Button
                          appearance="subtle"
                          size="small"
                          icon={<Delete24Regular />}
                          onClick={() => handleRemoveNode(node.node_id)}
                        >
                          移除
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </div>

      <div className={styles.bottomGrid}>
        <div className={styles.metricCard}>
          <Text weight="semibold" size={400}>任务监控</Text>
          <div className={styles.taskStatsGrid}>
            <div className={styles.taskStatItem}>
              <Text className={styles.taskStatLabel}>总数</Text>
              <Text className={styles.taskStatValue}>{taskStats.total}</Text>
            </div>
            <div className={styles.taskStatItem}>
              <Text className={styles.taskStatLabel}>完成</Text>
              <Text className={styles.taskStatValue}>{taskStats.completed}</Text>
            </div>
            <div className={styles.taskStatItem}>
              <Text className={styles.taskStatLabel}>失败</Text>
              <Text className={styles.taskStatValue}>{taskStats.failed}</Text>
            </div>
            <div className={styles.taskStatItem}>
              <Text className={styles.taskStatLabel}>待处理</Text>
              <Text className={styles.taskStatValue}>{taskStats.pending}</Text>
            </div>
          </div>
          <div className={styles.progressSection}>
            <div className={styles.progressLabel}>
              <Text size={200}>完成率</Text>
              <Text size={200}>{(completionRate * 100).toFixed(1)}%</Text>
            </div>
            <ProgressBar value={completionRate} thickness="large" />
          </div>
        </div>

        <div className={styles.metricCard}>
          <Text weight="semibold" size={400}>集群指标</Text>
          <div className={styles.metricList}>
            <div className={styles.metricRow}>
              <Text className={styles.metricLabel}>平均延迟</Text>
              <Text className={styles.metricValue}>
                {clusterMetricsData?.cluster_avg_latency != null && clusterMetricsData.cluster_avg_latency > 0
                  ? `${clusterMetricsData.cluster_avg_latency.toFixed(0)}ms`
                  : '-'}
              </Text>
            </div>
            <div className={styles.metricRow}>
              <Text className={styles.metricLabel}>吞吐量</Text>
              <Text className={styles.metricValue}>
                {clusterMetricsData?.cluster_throughput != null && clusterMetricsData.cluster_throughput > 0
                  ? `${clusterMetricsData.cluster_throughput.toFixed(1)}/s`
                  : '0.0/s'}
              </Text>
            </div>
            <div className={styles.metricRow}>
              <Text className={styles.metricLabel}>成功率</Text>
              <Text className={styles.metricValue}>
                {clusterMetricsData?.cluster_success_rate != null && clusterMetricsData.cluster_success_rate > 0
                  ? `${(clusterMetricsData.cluster_success_rate * 100).toFixed(1)}%`
                  : '-'}
              </Text>
            </div>
          </div>
        </div>
      </div>
        </>
      )}
    </div>
  )
}
