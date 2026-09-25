<script setup lang="ts">
// 仪表盘页面：系统健康总览 + 最近任务 + 快捷操作。
// 滚动由 AppLayout 内容区承担，页面只负责内容组织与状态分支。
import { onMounted } from 'vue'
import { PageHeader, StatusBadge, EmptyState, LoadingSkeleton } from '@/components'
import { systemApi, taskApi } from '@/api/client'
import type { SystemStatus, Task } from '@/api/types'
import { useApi } from '@/composables/useApi'
import HealthOverviewSection from '@/components/dashboard/HealthOverviewSection.vue'
import RecentTasksSection from '@/components/dashboard/RecentTasksSection.vue'
import QuickActionsSection from '@/components/dashboard/QuickActionsSection.vue'

onMounted(() => {
  document.title = '仪表盘 - PPC_NEXT'
})

// 系统状态：每 10s 轮询刷新
const {
  data: status,
  loading: statusLoading,
  error: statusError,
  refetch: refetchStatus,
} = useApi<SystemStatus>(() => systemApi.getStatus(), { refreshInterval: 10000 })

// 最近任务
const {
  data: tasks,
  loading: tasksLoading,
  error: tasksError,
  refetch: refetchTasks,
} = useApi<Task[]>(() => taskApi.list())
</script>

<template>
  <div class="ppc-page">
    <div class="ppc-page-container">
      <PageHeader title="仪表盘" description="系统运行状态概览">
        <template #actions>
          <StatusBadge
            v-if="status && !statusError"
            :status="
              status.status === 'running'
                ? 'running'
                : status.status === 'error'
                  ? 'failed'
                  : 'warning'
            "
          />
        </template>
      </PageHeader>

      <div class="ppc-dashboard-content">
        <!-- 首次加载骨架：健康总览区域 -->
        <template v-if="statusLoading && !status">
          <div class="ppc-dashboard-hero">
            <LoadingSkeleton variant="card" :height="'120px'" />
          </div>
          <div class="ppc-dashboard-grid">
            <LoadingSkeleton v-for="i in 4" :key="i" variant="card" />
          </div>
          <LoadingSkeleton variant="rect" :height="'200px'" />
        </template>

        <!-- 状态拉取失败：提示 + 重试 -->
        <EmptyState
          v-else-if="statusError"
          icon="\uE8F1"
          title="无法获取系统状态"
          :message="`无法获取系统状态: ${statusError}`"
          action-label="重试"
          :on-action="refetchStatus"
        />

        <!-- 状态就绪：渲染三大区块 -->
        <template v-else-if="status">
          <HealthOverviewSection :status="status" />

          <!-- 最近任务：任务为空 / 加载中均有对应状态 -->
          <div v-if="tasksLoading && !tasks" class="ppc-dashboard-tasks-skeleton">
            <LoadingSkeleton v-for="i in 3" :key="i" variant="card" />
          </div>
          <EmptyState
            v-else-if="tasksError"
            icon="\uE8F1"
            title="任务加载失败"
            :message="`无法获取任务列表: ${tasksError}`"
            action-label="重试"
            :on-action="refetchTasks"
          />
          <RecentTasksSection v-else :tasks="tasks" />

          <QuickActionsSection />
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ppc-dashboard-content {
  display: flex;
  flex-direction: column;
  gap: 24px;
}
.ppc-dashboard-hero {
  margin-bottom: 0;
}
.ppc-dashboard-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.ppc-dashboard-tasks-skeleton {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
</style>
