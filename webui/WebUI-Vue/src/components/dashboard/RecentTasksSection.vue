<script setup lang="ts">
// 最近任务列表区块：按创建时间倒序取前 5 条，带状态徽标。
// 对齐 React `src/components/dashboard/RecentTasksSection.tsx`。
import { computed } from 'vue'
import { WinTextBlock } from 'winui-web-library'
import { StatusBadge, EmptyState } from '@/components'
import type { Task, TaskStatus } from '@/api/types'

const props = defineProps<{ tasks: Task[] | null }>()

function formatDateTime(value: string | number): string {
  try {
    const d = typeof value === 'number' ? new Date(value * 1000) : new Date(value)
    const m = String(d.getMonth() + 1).padStart(2, '0')
    const day = String(d.getDate()).padStart(2, '0')
    const hh = String(d.getHours()).padStart(2, '0')
    const mm = String(d.getMinutes()).padStart(2, '0')
    return `${m}/${day} ${hh}:${mm}`
  } catch {
    return String(value)
  }
}

function formatTaskType(type: string): string {
  switch (type) {
    case 'tts':
    case 'convert':
      return 'TTS 转换'
    case 'batch':
      return '批量处理'
    case 'fanqie_install':
      return '番茄安装'
    default:
      return type
  }
}

function toStatusType(status: TaskStatus) {
  switch (status) {
    case 'running':
      return 'running'
    case 'completed':
      return 'completed'
    case 'failed':
      return 'failed'
    case 'cancelled':
      return 'cancelled'
    case 'pending':
    default:
      return 'pending'
  }
}

const recentTasks = computed(() => {
  if (!props.tasks) return []
  return [...props.tasks]
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
    .slice(0, 5)
})
</script>

<template>
  <section class="ppc-tasks-section">
    <WinTextBlock class="ppc-section-title" Text="最近任务" />
    <div v-if="recentTasks.length === 0" class="ppc-tasks-empty">
      <EmptyState message="暂无任务记录" />
    </div>
    <div v-else class="ppc-task-list">
      <div v-for="task in recentTasks" :key="task.task_id" class="ppc-task-row">
        <div class="ppc-task-left">
          <StatusBadge :status="toStatusType(task.status)" />
          <WinTextBlock class="ppc-task-label" :Text="formatTaskType(task.task_type)" />
        </div>
        <WinTextBlock class="ppc-task-meta" :Text="formatDateTime(task.created_at)" />
      </div>
    </div>
  </section>
</template>

<style scoped>
.ppc-tasks-section {
  display: flex;
  flex-direction: column;
}
.ppc-tasks-empty {
  background-color: var(--layer-default, #f3f3f3);
  border: 1px solid var(--ctrl-border, rgba(0, 0, 0, 0.08));
  border-radius: 9px;
}
.ppc-task-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.ppc-task-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 24px;
  background-color: var(--layer-default, #f3f3f3);
  border: 1px solid var(--ctrl-border, rgba(0, 0, 0, 0.08));
  border-radius: 9px;
  transition: background-color 167ms cubic-bezier(0.33, 0, 0.67, 1);
}
.ppc-task-row:hover {
  background-color: var(--card-bg, #e9e9e9);
}
.ppc-task-left {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
  flex: 1;
}
.ppc-task-label {
  color: var(--text-primary, #000);
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ppc-task-meta {
  color: var(--text-secondary, #555);
  font-size: 12px;
  white-space: nowrap;
  flex-shrink: 0;
}
</style>