<script setup lang="ts">
// 页头组件：标题 + 描述 + breadcrumb/actions 插槽。
// 宽度由页面的 .ppc-page-container 约束，保证分隔线与正文左缘对齐。
import { WinTextBlock } from 'winui-web-library'

defineProps<{
  title: string
  description?: string
}>()
</script>

<template>
  <div class="ppc-page-header">
    <div class="ppc-page-header-group">
      <div v-if="$slots.breadcrumb" class="ppc-page-header-breadcrumb">
        <slot name="breadcrumb" />
      </div>
      <WinTextBlock class="ppc-page-header-title" FontSize="20" FontWeight="600" :Text="title" />
      <WinTextBlock v-if="description" class="ppc-page-header-desc" FontSize="13" :Text="description" />
    </div>
    <div v-if="$slots.actions" class="ppc-page-header-actions">
      <slot name="actions" />
    </div>
  </div>
</template>

<style scoped>
.ppc-page-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  padding: 20px 0 16px;
  border-bottom: 1px solid var(--ctrl-border-accent, rgba(0, 0, 0, 0.16));
  margin-bottom: 20px;
}
.ppc-page-header-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  min-width: 0;
}
.ppc-page-header-breadcrumb {
  font-size: 12px;
  color: var(--text-tertiary, #777);
}
.ppc-page-header-title {
  color: var(--text-primary, #000);
  line-height: 1.3;
}
.ppc-page-header-desc {
  color: var(--text-secondary, #555);
  max-width: 64ch;
}
.ppc-page-header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
  padding-bottom: 4px;
}
</style>
