<script setup lang="ts">
// 健康总览区块：健康评分环 + 资源使用率卡片 + 运行信息。
// 对齐 React `src/components/dashboard/HealthOverviewSection.tsx`。
import { computed } from 'vue'
import { WinTextBlock, WinProgressBar } from 'winui-web-library'
import type { SystemStatus } from '@/api/types'
import { designTokens } from '@/theme/tokens'

const props = defineProps<{ status: SystemStatus }>()

const CIRCLE_RADIUS = 52
const CIRCLE_CIRCUMFERENCE = 2 * Math.PI * CIRCLE_RADIUS
const CIRCLE_SIZE = (CIRCLE_RADIUS + 8) * 2

function getHealthColor(score: number): string {
  if (score >= 90) return designTokens.status.success
  if (score >= 70) return designTokens.status.warning
  return designTokens.status.danger
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

const healthScore = computed(() => Math.round(props.status.health_score ?? 0))
const healthColor = computed(() => getHealthColor(healthScore.value))
const healthLabel = computed(() => getHealthLabel(healthScore.value))
const cpuPercent = computed(() => props.status.cpu_percent ?? 0)
const memoryPercent = computed(() => props.status.memory_percent ?? 0)
const diskPercent = computed(() => props.status.disk_percent ?? 0)
const strokeDashoffset = computed(() => CIRCLE_CIRCUMFERENCE * (1 - healthScore.value / 100))
const uptimeLabel = computed(() => formatUptime(props.status.uptime ?? 0))

// 资源条前景色：依据占用率映射到语义色
const barColor = (p: number) =>
  p >= 90 ? designTokens.status.danger : p >= 70 ? designTokens.status.warning : designTokens.status.success
</script>

<template>
  <div class="ppc-health-top">
    <!-- 健康评分面板 -->
    <div class="ppc-health-panel">
      <div class="ppc-health-circle">
        <svg
          class="ppc-health-svg"
          :width="CIRCLE_SIZE"
          :height="CIRCLE_SIZE"
          :viewBox="`0 0 ${CIRCLE_SIZE} ${CIRCLE_SIZE}`"
          role="img"
          :aria-label="`健康评分: ${healthScore}`"
        >
          <circle
            class="ppc-health-circle-bg"
            :cx="CIRCLE_SIZE / 2"
            :cy="CIRCLE_SIZE / 2"
            :r="CIRCLE_RADIUS"
          />
          <circle
            class="ppc-health-circle-fill"
            :cx="CIRCLE_SIZE / 2"
            :cy="CIRCLE_SIZE / 2"
            :r="CIRCLE_RADIUS"
            :style="{ stroke: healthColor }"
            :stroke-dasharray="CIRCLE_CIRCUMFERENCE"
            :stroke-dashoffset="strokeDashoffset"
          />
        </svg>
        <div class="ppc-health-score-text">
          <WinTextBlock
            class="ppc-health-score"
            FontSize="34"
            FontWeight="600"
            :Foreground="healthColor"
            :Text="String(healthScore)"
          />
          <WinTextBlock class="ppc-health-status" FontWeight="600" :Foreground="healthColor" :Text="healthLabel" />
        </div>
      </div>
      <WinTextBlock class="ppc-health-subtext" :Text="'系统健康评分'" />
    </div>

    <!-- 资源指标卡片 -->
    <div class="ppc-resource-grid">
      <div class="ppc-resource-tile">
        <div class="ppc-resource-header">
          <WinTextBlock class="ppc-resource-icon" FontFamily="WinUIOnWebIcons" FontSize="20" Text="&#xE7F8;" />
          <WinTextBlock class="ppc-resource-value" FontSize="22" FontWeight="600" :Text="`${cpuPercent.toFixed(1)}%`" />
        </div>
        <WinTextBlock class="ppc-resource-label" Text="CPU 使用率" />
        <WinProgressBar class="ppc-resource-progress" :Value="cpuPercent" :Maximum="100" :Foreground="barColor(cpuPercent)" />
      </div>

      <div class="ppc-resource-tile">
        <div class="ppc-resource-header">
          <WinTextBlock class="ppc-resource-icon" FontFamily="WinUIOnWebIcons" FontSize="20" Text="&#xE9D9;" />
          <WinTextBlock class="ppc-resource-value" FontSize="22" FontWeight="600" :Text="`${memoryPercent.toFixed(1)}%`" />
        </div>
        <WinTextBlock class="ppc-resource-label" Text="内存使用率" />
        <WinProgressBar class="ppc-resource-progress" :Value="memoryPercent" :Maximum="100" :Foreground="barColor(memoryPercent)" />
      </div>

      <div class="ppc-resource-tile">
        <div class="ppc-resource-header">
          <WinTextBlock class="ppc-resource-icon" FontFamily="WinUIOnWebIcons" FontSize="20" Text="&#xE8E0;" />
          <WinTextBlock class="ppc-resource-value" FontSize="22" FontWeight="600" :Text="`${diskPercent.toFixed(1)}%`" />
        </div>
        <WinTextBlock class="ppc-resource-label" Text="磁盘使用率" />
        <WinProgressBar class="ppc-resource-progress" :Value="diskPercent" :Maximum="100" :Foreground="barColor(diskPercent)" />
      </div>

      <div class="ppc-resource-tile">
        <div class="ppc-resource-header">
          <WinTextBlock class="ppc-resource-icon" FontFamily="WinUIOnWebIcons" FontSize="20" Text="&#xE943;" />
        </div>
        <div class="ppc-process-row">
          <WinTextBlock class="ppc-process-label" Text="版本" />
          <WinTextBlock class="ppc-process-value" FontWeight="600" :Text="status.version" />
        </div>
        <div class="ppc-process-row">
          <WinTextBlock class="ppc-process-label" Text="活跃任务" />
          <WinTextBlock class="ppc-process-value" FontWeight="600" :Text="String(status.active_tasks)" />
        </div>
        <div class="ppc-process-row">
          <WinTextBlock class="ppc-process-label" Text="运行时间" />
          <WinTextBlock class="ppc-process-value" FontWeight="600" :Text="uptimeLabel" />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ppc-health-top {
  display: grid;
  grid-template-columns: 1fr 1.5fr;
  gap: 32px;
}
@media (max-width: 1023px) {
  .ppc-health-top {
    grid-template-columns: 1fr;
  }
}

.ppc-health-panel {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 24px;
  background-color: var(--layer-default, #f3f3f3);
  border: 1px solid var(--ctrl-border, rgba(0, 0, 0, 0.08));
  border-radius: 9px;
}
.ppc-health-circle {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}
.ppc-health-svg {
  transform: rotate(-90deg);
}
.ppc-health-circle-bg {
  fill: none;
  stroke: var(--card-bg, #ececec);
  stroke-width: 8;
}
.ppc-health-circle-fill {
  fill: none;
  stroke-width: 8;
  stroke-linecap: round;
  transition:
    stroke-dashoffset 300ms cubic-bezier(0.33, 0, 0.67, 1),
    stroke 250ms cubic-bezier(0.33, 0, 0.67, 1);
}
.ppc-health-score-text {
  position: absolute;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}
.ppc-health-subtext {
  color: var(--text-secondary, #555);
}
.ppc-resource-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 16px;
}
@media (max-width: 500px) {
  .ppc-resource-grid {
    grid-template-columns: 1fr;
    grid-template-rows: auto;
  }
}
.ppc-resource-tile {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 20px 24px;
  background-color: var(--layer-default, #f3f3f3);
  border: 1px solid var(--ctrl-border, rgba(0, 0, 0, 0.08));
  border-radius: 9px;
  transition:
    transform 167ms cubic-bezier(0.33, 0, 0.67, 1),
    box-shadow 167ms cubic-bezier(0.33, 0, 0.67, 1);
}
.ppc-resource-tile:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
}
@media (prefers-reduced-motion: reduce) {
  .ppc-resource-tile {
    transform: none;
    transition: none;
  }
}
.ppc-resource-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.ppc-resource-icon {
  color: var(--accent-text, var(--accent-base));
  flex-shrink: 0;
  line-height: 1;
}
.ppc-resource-value {
  color: var(--text-primary, #000);
}
.ppc-resource-label {
  color: var(--text-secondary, #555);
}
.ppc-resource-progress {
  width: 100%;
  margin-top: 8px;
}
.ppc-process-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.ppc-process-label {
  color: var(--text-secondary, #555);
}
.ppc-process-value {
  color: var(--text-primary, #000);
}
</style>