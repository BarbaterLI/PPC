<script setup lang="ts">
// 状态徽标：running/completed/failed/pending/cancelled/warning + 尺寸。
import { computed } from 'vue'
import { WinTextBlock } from 'winui-web-library'

export type StatusType = 'running' | 'completed' | 'failed' | 'pending' | 'cancelled' | 'warning'
export type StatusBadgeSize = 'small' | 'medium' | 'large'

const props = withDefaults(
  defineProps<{
    status: StatusType
    label?: string
    size?: StatusBadgeSize
  }>(),
  { size: 'medium' },
)

const statusLabelMap: Record<StatusType, string> = {
  running: '运行中',
  completed: '已完成',
  failed: '失败',
  pending: '等待中',
  cancelled: '已取消',
  warning: '警告',
}

const badgeClass = computed(() => `ppc-badge--${props.status}`)
const sizeClass = computed(() => `ppc-badge--${props.size}`)
const text = computed(() => props.label ?? statusLabelMap[props.status])
</script>

<template>
  <WinTextBlock class="ppc-badge" :class="[badgeClass, sizeClass]" :Text="text" />
</template>

<style scoped>
.ppc-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 0 10px;
  border-radius: 6px;
  white-space: nowrap;
  user-select: none;
  font-weight: 500;
  line-height: 1;
}
.ppc-badge::before {
  content: '';
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.ppc-badge--small {
  height: 20px;
  font-size: 11px;
}
.ppc-badge--medium {
  height: 24px;
  font-size: 12px;
}
.ppc-badge--large {
  height: 28px;
  font-size: 14px;
}

.ppc-badge--running {
  background: var(--ppc-status-success-bg, rgba(15, 123, 15, 0.12));
  color: var(--ppc-status-success, #0f7b0f);
}
.ppc-badge--running::before {
  background: var(--ppc-status-success, #0f7b0f);
  animation: ppc-dot-pulse 1.6s linear infinite;
}
.ppc-badge--completed {
  background: var(--ppc-status-success-bg, rgba(15, 123, 15, 0.12));
  color: var(--ppc-status-success, #0f7b0f);
}
.ppc-badge--completed::before {
  background: var(--ppc-status-success, #0f7b0f);
}
.ppc-badge--failed {
  background: var(--ppc-status-danger-bg, rgba(196, 43, 28, 0.12));
  color: var(--ppc-status-danger, #c42b1c);
}
.ppc-badge--failed::before {
  background: var(--ppc-status-danger, #c42b1c);
}
.ppc-badge--pending {
  background: rgba(128, 128, 128, 0.1);
  color: var(--text-secondary, #555);
}
.ppc-badge--pending::before {
  background: var(--text-tertiary, #999);
}
.ppc-badge--cancelled,
.ppc-badge--warning {
  background: var(--ppc-status-warning-bg, rgba(157, 93, 0, 0.14));
  color: var(--ppc-status-warning, #9d5d00);
}
.ppc-badge--cancelled::before,
.ppc-badge--warning::before {
  background: var(--ppc-status-warning, #9d5d00);
}

@keyframes ppc-dot-pulse {
  0%,
  100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.45;
    transform: scale(0.8);
  }
}

@media (prefers-reduced-motion: reduce) {
  .ppc-badge--running::before {
    animation: none;
  }
}
</style>