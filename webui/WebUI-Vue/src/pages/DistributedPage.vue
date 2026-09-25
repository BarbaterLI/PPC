<script setup lang="ts">
// 分布式管理页 —— 对齐 React `pages/DistributedPage.tsx`。
// 功能：集群总览 / 调度器与本机节点控制 / 添加节点 / 节点表(排空/激活/移除) / 任务监控 / 集群指标。
import { onMounted, ref, watch } from 'vue'
import {
  WinButton,
  WinContentDialog,
  WinGrid,
  WinInfoBar,
  WinNumberBox,
  WinProgressBar,
  WinStackPanel,
  WinTextBox,
  WinTextBlock,
} from 'winui-web-library'
import { PageHeader, StatusBadge, ConfirmDialog, LoadingSkeleton } from '@/components'
import { useApi } from '@/composables/useApi'
import { distributedApi } from '@/api/client'
import type { ClusterStatus, ClusterMetricsResponse, NodeInfo } from '@/api/types'
import type { StatusType } from '@/components/StatusBadge.vue'

onMounted(() => {
  document.title = '分布式管理 - PPC_NEXT'
})

interface ConfirmPending {
  nodeId: string
  action: 'remove' | 'drain'
}

// ---- 数据（统一 5s 轮询，对齐 React 的单一定时器语义） ----
const POLL_MS = 5000
const { data: clusterStatus, loading: statusLoading, refetch: refetchStatus } = useApi<ClusterStatus>(
  () => distributedApi.getStatus(),
  { refreshInterval: POLL_MS },
)
const { data: nodes, loading: nodesLoading, refetch: refetchNodes } = useApi<NodeInfo[]>(
  () => distributedApi.getNodes(),
  { refreshInterval: POLL_MS },
)
const { data: metrics, refetch: refetchMetrics } = useApi<ClusterMetricsResponse>(
  () => distributedApi.getMetrics(),
  { refreshInterval: POLL_MS },
)

// ---- 操作状态 ----
const addDialogOpen = ref(false)
const newHost = ref('')
const newPort = ref('8080')
const newMaxConcurrency = ref('4')
const operating = ref<string | null>(null)
const lastError = ref<string | null>(null)

const schedulerRunning = ref(false)
const nodeServiceRunning = ref(false)
watch(clusterStatus, (st) => {
  schedulerRunning.value = st?.running ?? false
  nodeServiceRunning.value = st?.node_service_running ?? false
}, { immediate: true })

const confirmDialogRef = ref<InstanceType<typeof ConfirmDialog> | null>(null)

// ---- 指令辅助 ----
const refetchAll = () => {
  refetchStatus()
  refetchNodes()
  refetchMetrics()
}

const runAction = async (label: string, action: () => Promise<unknown>) => {
  operating.value = label
  lastError.value = null
  try {
    await action()
    refetchAll()
  } catch (err: unknown) {
    lastError.value = err instanceof Error ? err.message : '网络错误'
  } finally {
    operating.value = null
  }
}

const handleStartScheduler = () => {
  runAction('scheduler-start', async () => {
    const r = await distributedApi.startScheduler()
    if (r.success === false) throw new Error(r.error?.message || '启动失败')
  })
}

const handleStopScheduler = () => {
  runAction('scheduler-stop', async () => {
    const r = await distributedApi.stopScheduler()
    if (r.success === false) throw new Error(r.error?.message || '停止失败')
  })
}

const handleStartNodeService = () => {
  runAction('node-start', async () => {
    const r = await distributedApi.startNodeService()
    if (r.success === false) throw new Error(r.error?.message || '启动失败')
  })
}

const handleStopNodeService = () => {
  runAction('node-stop', async () => {
    const r = await distributedApi.stopNodeService()
    if (r.success === false) throw new Error(r.error?.message || '停止失败')
  })
}

const handleAddNode = () => {
  const port = parseInt(newPort.value, 10)
  const maxConcurrency = parseInt(newMaxConcurrency.value, 10)
  if (!newHost.value || Number.isNaN(port) || Number.isNaN(maxConcurrency)) return

  runAction('add-node', async () => {
    const r = await distributedApi.addNode(newHost.value, port, maxConcurrency)
    if (r.success === false) throw new Error(r.error?.message || '添加节点失败')
    addDialogOpen.value = false
    newHost.value = ''
    newPort.value = '8080'
    newMaxConcurrency.value = '4'
  })
}

const performNodeAction = async (pending: ConfirmPending) => {
  await runAction(pending.action, async () => {
    const r =
      pending.action === 'remove'
        ? await distributedApi.removeNode(pending.nodeId)
        : await distributedApi.drainNode(pending.nodeId)
    if (r.success === false) {
      throw new Error(r.error?.message || (pending.action === 'remove' ? '移除节点失败' : '排空节点失败'))
    }
  })
}

const handleRemoveNode = async (nodeId: string) => {
  if (!confirmDialogRef.value) return
  const ok = await confirmDialogRef.value.confirm({
    title: '移除节点',
    message: `确定要移除节点 ${nodeId} 吗？`,
    okText: '移除',
    cancelText: '取消',
    danger: true,
  })
  if (ok) performNodeAction({ nodeId, action: 'remove' })
}

const handleDrainNode = async (nodeId: string) => {
  if (!confirmDialogRef.value) return
  const ok = await confirmDialogRef.value.confirm({
    title: '排空节点',
    message: `确定要排空节点 ${nodeId} 吗？`,
    okText: '排空',
    cancelText: '取消',
    danger: true,
  })
  if (ok) performNodeAction({ nodeId, action: 'drain' })
}

const handleActivateNode = (nodeId: string) => {
  runAction('activate', async () => {
    const r = await distributedApi.activateNode(nodeId)
    if (r.success === false) throw new Error(r.error?.message || '激活节点失败')
  })
}

// ---- 节点状态映射 ----
function nodeStatusType(status: NodeInfo['status']): StatusType {
  switch (status) {
    case 'active':
      return 'running'
    case 'inactive':
      return 'pending'
    case 'unhealthy':
      return 'failed'
    case 'draining':
      return 'warning'
  }
}

function getStatusBadgeLabel(status: NodeInfo['status']): string {
  switch (status) {
    case 'active':
      return '活跃'
    case 'inactive':
      return '离线'
    case 'unhealthy':
      return '异常'
    case 'draining':
      return '排空中'
  }
}
</script>

<template>
  <div class="ppc-page">
    <div class="ppc-page-container">
      <PageHeader title="分布式" description="分布式集群管理" />

      <div class="ppc-dist-content">
      <LoadingSkeleton v-if="(statusLoading || nodesLoading) && !clusterStatus && !nodes" variant="card" :count="3" />

      <template v-else>
        <WinInfoBar
          v-if="lastError"
          Severity="Error"
          :IsOpen="Boolean(lastError)"
          Title="错误"
          :Message="lastError"
        />

        <!-- 集群总览 -->
        <WinGrid ColumnSpacing="16" RowSpacing="16" ColumnDefinitions="1fr,1fr,1fr,1fr">
          <div class="ppc-card ppc-card--compact">
            <WinTextBlock class="ppc-stat-label" FontSize="12" :Text="'节点总数'" />
            <WinTextBlock class="ppc-stat-value" FontSize="22" FontWeight="600">
              {{ clusterStatus?.nodes?.total ?? 0 }}
            </WinTextBlock>
          </div>
          <div class="ppc-card ppc-card--compact">
            <WinTextBlock class="ppc-stat-label" FontSize="12" :Text="'活跃节点'" />
            <WinTextBlock class="ppc-stat-value ppc-stat-accent" FontSize="22" FontWeight="600">
              {{ clusterStatus?.nodes?.active ?? 0 }}
            </WinTextBlock>
          </div>
          <div class="ppc-card ppc-card--compact">
            <WinTextBlock class="ppc-stat-label" FontSize="12" :Text="'集群吞吐量'" />
            <WinTextBlock class="ppc-stat-value" FontSize="22" FontWeight="600">
              {{
                metrics?.cluster?.cluster_throughput && metrics.cluster.cluster_throughput > 0
                  ? `${metrics.cluster.cluster_throughput.toFixed(1)}/s`
                  : '0.0/s'
              }}
            </WinTextBlock>
          </div>
          <div class="ppc-card ppc-card--compact">
            <WinTextBlock class="ppc-stat-label" FontSize="12" :Text="'平均延迟'" />
            <WinTextBlock class="ppc-stat-value" FontSize="22" FontWeight="600">
              {{
                metrics?.cluster?.cluster_avg_latency && metrics.cluster.cluster_avg_latency > 0
                  ? `${metrics.cluster.cluster_avg_latency.toFixed(0)}ms`
                  : '-'
              }}
            </WinTextBlock>
          </div>
        </WinGrid>

        <!-- 控制区 -->
        <div class="ppc-action-bar">
          <WinButton
            v-if="schedulerRunning"
            Style="AccentButtonStyle"
            :IsEnabled="operating !== 'scheduler-stop'"
            :Content="'停止调度器'"
            @Click="handleStopScheduler"
          />
          <WinButton
            v-else
            Style="AccentButtonStyle"
            :IsEnabled="operating !== 'scheduler-start'"
            :Content="'启动调度器'"
            @Click="handleStartScheduler"
          />
          <WinButton
            v-if="nodeServiceRunning"
            Style="DefaultButtonStyle"
            :IsEnabled="operating !== 'node-stop'"
            :Content="'停止本机节点'"
            @Click="handleStopNodeService"
          />
          <WinButton
            v-else
            Style="DefaultButtonStyle"
            :IsEnabled="operating !== 'node-start'"
            :Content="'启动本机节点'"
            @Click="handleStartNodeService"
          />
          <WinButton
            Style="DefaultButtonStyle"
            :IsEnabled="Boolean(schedulerRunning)"
            :Content="'+ 添加节点'"
            @Click="addDialogOpen = true"
          />
        </div>

        <!-- 节点表 -->
        <div>
          <WinTextBlock class="ppc-section-title" FontSize="16" FontWeight="600" :Text="'节点管理'" />
          <div class="ppc-node-table-wrap">
            <table class="ppc-node-table">
              <thead>
                <tr>
                  <th>节点 ID</th>
                  <th>地址</th>
                  <th>状态</th>
                  <th>并发</th>
                  <th>成功率</th>
                  <th>平均延迟</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                <tr v-if="!nodes || nodes.length === 0">
                  <td colspan="7">
                    <div class="ppc-node-empty">
                      <WinTextBlock FontSize="12" class="ppc-node-empty-text">
                        {{ schedulerRunning ? '暂无节点数据，点击"添加节点"添加远程节点' : '请先启动调度器' }}
                      </WinTextBlock>
                    </div>
                  </td>
                </tr>
                <tr v-for="node in nodes" :key="node.node_id">
                  <td>
                    <WinTextBlock FontSize="12" FontFamily="Consolas, monospace" :Text="node.node_id" />
                  </td>
                  <td>
                    <WinTextBlock FontSize="12" FontFamily="Consolas, monospace">
                      {{ node.host }}:{{ node.port }}
                    </WinTextBlock>
                  </td>
                  <td>
                    <StatusBadge :status="nodeStatusType(node.status)" :label="getStatusBadgeLabel(node.status)" size="small" />
                  </td>
                  <td>
                    <WinTextBlock FontSize="12">
                      {{ node.current_concurrency }}/{{ node.max_concurrency }}
                    </WinTextBlock>
                  </td>
                  <td>
                    <WinTextBlock FontSize="12">{{ node.success_rate.toFixed(1) }}%</WinTextBlock>
                  </td>
                  <td>
                    <WinTextBlock FontSize="12">{{ node.avg_response_time.toFixed(0) }}ms</WinTextBlock>
                  </td>
                  <td>
                    <div class="ppc-action-cell">
                      <WinButton
                        v-if="node.status === 'active'"
                        Style="SubtleButtonStyle"
                        :Content="'排空'"
                        @Click="handleDrainNode(node.node_id)"
                      />
                      <WinButton
                        v-if="node.status === 'inactive' || node.status === 'draining'"
                        Style="SubtleButtonStyle"
                        :Content="'激活'"
                        @Click="handleActivateNode(node.node_id)"
                      />
                      <WinButton Style="SubtleButtonStyle" :Content="'移除'" @Click="handleRemoveNode(node.node_id)" />
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- 任务监控 + 集群指标 -->
        <WinGrid ColumnSpacing="16" RowSpacing="16" ColumnDefinitions="1fr,1fr">
          <div class="ppc-card">
            <WinTextBlock FontSize="16" FontWeight="600" :Text="'任务监控'" />
            <WinGrid ColumnSpacing="16" RowSpacing="12" ColumnDefinitions="1fr,1fr">
              <div class="ppc-task-stat">
                <WinTextBlock class="ppc-stat-label" FontSize="12" :Text="'总数'" />
                <WinTextBlock class="ppc-stat-value" FontSize="18" FontWeight="600">
                  {{ clusterStatus?.tasks?.total ?? 0 }}
                </WinTextBlock>
              </div>
              <div class="ppc-task-stat">
                <WinTextBlock class="ppc-stat-label" FontSize="12" :Text="'完成'" />
                <WinTextBlock class="ppc-stat-value ppc-stat-success" FontSize="18" FontWeight="600">
                  {{ clusterStatus?.tasks?.completed ?? 0 }}
                </WinTextBlock>
              </div>
              <div class="ppc-task-stat">
                <WinTextBlock class="ppc-stat-label" FontSize="12" :Text="'失败'" />
                <WinTextBlock class="ppc-stat-value ppc-stat-failure" FontSize="18" FontWeight="600">
                  {{ clusterStatus?.tasks?.failed ?? 0 }}
                </WinTextBlock>
              </div>
              <div class="ppc-task-stat">
                <WinTextBlock class="ppc-stat-label" FontSize="12" :Text="'待处理'" />
                <WinTextBlock class="ppc-stat-value" FontSize="18" FontWeight="600">
                  {{ clusterStatus?.tasks?.pending ?? 0 }}
                </WinTextBlock>
              </div>
            </WinGrid>
            <div class="ppc-progress">
              <div class="ppc-progress-label">
                <WinTextBlock FontSize="12" :Text="'完成率'" />
                <WinTextBlock FontSize="12">
                  {{
                    ((clusterStatus?.tasks?.total ?? 0) > 0
                      ? (clusterStatus!.tasks.completed / clusterStatus!.tasks.total) * 100
                      : 0
                    ).toFixed(1)
                  }}%
                </WinTextBlock>
              </div>
              <WinProgressBar
                :Value="(clusterStatus?.tasks?.total ?? 0) > 0 ? clusterStatus!.tasks.completed / clusterStatus!.tasks.total : 0"
                :Maximum="1"
                :Width="'100%'"
                :Height="4"
              />
            </div>
          </div>

          <div class="ppc-card">
            <WinTextBlock FontSize="16" FontWeight="600" :Text="'集群指标'" />
            <WinStackPanel Spacing="14">
              <div class="ppc-metric-row">
                <WinTextBlock class="ppc-stat-label" FontSize="14" :Text="'平均延迟'" />
                <WinTextBlock class="ppc-stat-value" FontSize="14" FontWeight="600">
                  {{
                    metrics?.cluster?.cluster_avg_latency && metrics.cluster.cluster_avg_latency > 0
                      ? `${metrics.cluster.cluster_avg_latency.toFixed(0)}ms`
                      : '-'
                  }}
                </WinTextBlock>
              </div>
              <div class="ppc-metric-row">
                <WinTextBlock class="ppc-stat-label" FontSize="14" :Text="'吞吐量'" />
                <WinTextBlock class="ppc-stat-value" FontSize="14" FontWeight="600">
                  {{
                    metrics?.cluster?.cluster_throughput && metrics.cluster.cluster_throughput > 0
                      ? `${metrics.cluster.cluster_throughput.toFixed(1)}/s`
                      : '0.0/s'
                  }}
                </WinTextBlock>
              </div>
              <div class="ppc-metric-row">
                <WinTextBlock class="ppc-stat-label" FontSize="14" :Text="'成功率'" />
                <WinTextBlock class="ppc-stat-value" FontSize="14" FontWeight="600">
                  {{
                    metrics?.cluster?.cluster_success_rate && metrics.cluster.cluster_success_rate > 0
                      ? `${(metrics.cluster.cluster_success_rate * 100).toFixed(1)}%`
                      : '-'
                  }}
                </WinTextBlock>
              </div>
            </WinStackPanel>
          </div>
        </WinGrid>
      </template>
      </div>
    </div>

    <!-- 添加节点对话框 -->
    <WinContentDialog
      :IsOpen="addDialogOpen"
      :Title="'添加节点'"
      :PrimaryButtonText="'添加'"
      :SecondaryButtonText="'取消'"
      :IsPrimaryButtonEnabled="operating !== 'add-node'"
      @PrimaryButtonClick="handleAddNode"
      @update:IsOpen="(v: boolean) => (addDialogOpen = v)"
    >
      <WinStackPanel Spacing="14">
        <div class="ppc-dialog-field">
          <WinTextBlock class="ppc-dialog-label" FontSize="12" :Text="'主机地址'" />
          <WinTextBox
            :Text="newHost"
            PlaceholderText="例如: 192.168.1.100"
            @update:Text="(v: string) => (newHost = v)"
          />
        </div>
        <div class="ppc-dialog-field">
          <WinTextBlock class="ppc-dialog-label" FontSize="12" :Text="'端口'" />
          <WinNumberBox
            :Value="parseInt(newPort, 10) || 8080"
            :Minimum="1"
            :Maximum="65535"
            :SmallChange="1"
            @update:Value="(v: number) => (newPort = Number.isNaN(v) ? '' : String(v))"
          />
        </div>
        <div class="ppc-dialog-field">
          <WinTextBlock class="ppc-dialog-label" FontSize="12" :Text="'最大并发数'" />
          <WinNumberBox
            :Value="parseInt(newMaxConcurrency, 10) || 4"
            :Minimum="1"
            :Maximum="1024"
            :SmallChange="1"
            @update:Value="(v: number) => (newMaxConcurrency = Number.isNaN(v) ? '' : String(v))"
          />
        </div>
      </WinStackPanel>
    </WinContentDialog>

    <!-- 排空 / 移除 确认 -->
    <ConfirmDialog ref="confirmDialogRef" />
  </div>
</template>

<style scoped>
.ppc-dist-content {
  display: flex;
  flex-direction: column;
  gap: 24px;
  width: 100%;
}
.ppc-stat-label {
  color: var(--text-secondary, #555);
}
.ppc-stat-value {
  color: var(--text-primary, #000);
}
.ppc-stat-accent {
  color: var(--accent-text, #bf5b3b);
}
.ppc-stat-success {
  color: var(--ppc-status-success, #0f7b0f);
}
.ppc-stat-failure {
  color: var(--ppc-status-danger, #c42b1c);
}
.ppc-action-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.ppc-node-table-wrap {
  overflow-x: auto;
  background-color: var(--layer-default, #fff);
  border: 1px solid var(--ctrl-border, rgba(0, 0, 0, 0.16));
  border-radius: 9px;
}
.ppc-node-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.ppc-node-table th,
.ppc-node-table td {
  padding: 10px 14px;
  text-align: left;
  border-bottom: 1px solid var(--ctrl-border, rgba(0, 0, 0, 0.08));
  color: var(--text-primary, #000);
  white-space: nowrap;
}
.ppc-node-table th {
  background-color: var(--layer-default, #f9f9f9);
  font-weight: 600;
  color: var(--text-secondary, #555);
}
.ppc-node-table tbody tr:hover {
  background-color: var(--subtle-fill-secondary, rgba(0, 0, 0, 0.03));
}
.ppc-node-table tbody tr:last-child td {
  border-bottom: none;
}
.ppc-node-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 32px 0;
}
.ppc-node-empty-text {
  color: var(--text-secondary, #555);
}
.ppc-action-cell {
  display: flex;
  align-items: center;
  gap: 4px;
}
.ppc-task-stat {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.ppc-progress {
  margin-top: 16px;
}
.ppc-progress-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}
.ppc-metric-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.ppc-dialog-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.ppc-dialog-label {
  color: var(--text-secondary, #555);
}
</style>