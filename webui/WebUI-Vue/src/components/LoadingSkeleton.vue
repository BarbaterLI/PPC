<script setup lang="ts">
// 骨架屏：文本/矩形/圆/卡片/行等变体，支持 count 与自定义尺寸。
import { computed } from 'vue'

type SkeletonVariant = 'text' | 'rect' | 'circle' | 'card' | 'line'

const props = withDefaults(
  defineProps<{
    variant?: SkeletonVariant
    width?: string
    height?: string
    count?: number
    className?: string
  }>(),
  { variant: 'text', count: 1 },
)

const count = computed(() => Math.max(1, props.count || 1))
const style = computed<Record<string, string>>(() => {
  const s: Record<string, string> = {}
  if (props.width) s.width = props.width
  if (props.height) s.height = props.height
  return s
})
</script>

<template>
  <div v-if="count > 1" class="ppc-skeleton-list">
    <div
      v-for="i in count"
      :key="i"
      class="ppc-skeleton"
      :class="[`ppc-skeleton--${variant}`]"
      :style="style"
    />
  </div>
  <div v-else class="ppc-skeleton" :class="[`ppc-skeleton--${variant}`]" :style="style" />
</template>

<style scoped>
.ppc-skeleton {
  background: linear-gradient(
    90deg,
    var(--card-bg, #ececec) 25%,
    var(--layer-default, #f7f7f7) 50%,
    var(--card-bg, #ececec) 75%
  );
  background-size: 200% 100%;
  animation: ppc-shimmer 1.6s linear infinite;
}
.ppc-skeleton-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.ppc-skeleton--text {
  width: 100%;
  height: 12px;
  border-radius: 3px;
}
.ppc-skeleton--line {
  width: 100%;
  height: 8px;
  border-radius: 3px;
}
.ppc-skeleton--rect {
  width: 100%;
  height: 32px;
  border-radius: 6px;
}
.ppc-skeleton--card {
  width: 100%;
  height: 80px;
  border-radius: 9px;
}
.ppc-skeleton--circle {
  width: 40px;
  height: 40px;
  border-radius: 50%;
}

@keyframes ppc-shimmer {
  from {
    background-position: 120% 0;
  }
  to {
    background-position: -160% 0;
  }
}

@media (prefers-reduced-motion: reduce) {
  .ppc-skeleton {
    animation: none;
  }
}
</style>