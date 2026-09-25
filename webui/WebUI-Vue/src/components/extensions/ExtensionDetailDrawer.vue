<script setup lang="ts">
// 扩展详情抽屉 —— 复刻 React 工程 `src/components/extensions/ExtensionDetailDrawer.tsx`。
// 库暂无 Drawer 控件，改用右侧滑出面板 + 背景遮罩实现 WinUI 抽屉交互。
import { computed } from 'vue'
import { WinButton, WinTextBlock, WinToggleSwitch } from 'winui-web-library'
import { getExtensionTypeLabel } from './extensionType'
import type { Extension } from '@/api/types'

const props = defineProps<{
  open: boolean
  ext: Extension | null
  toggling: boolean
  onOpenChange: (open: boolean) => void
  onToggle: (ext: Extension) => void
}>()

const accentColor = computed(() => 'var(--accent-text, #53afab)')

const configEntries = computed(() => {
  const cfg = props.ext?.config
  if (!cfg || typeof cfg !== 'object') return []
  return Object.entries(cfg as Record<string, unknown>)
})

const formatValue = (value: unknown): string =>
  typeof value === 'string' ? value : JSON.stringify(value)
</script>

<template>
  <Teleport to="body">
    <Transition name="ppc-drawer" :duration="{ enter: 250, leave: 167 }">
      <div v-if="open" class="ppc-drawer-root">
        <div class="ppc-drawer-backdrop" @pointerdown="onOpenChange(false)" @click="onOpenChange(false)" />
        <aside class="ppc-drawer-panel" role="dialog" aria-modal="true">
          <header class="ppc-drawer-header">
            <div class="ppc-drawer-title-group">
              <span class="ppc-drawer-icon" :style="{ color: accentColor }">
                {{ ext?.has_webui ? '\uE774' : '\uE8B3' }}
              </span>
              <WinTextBlock FontSize="18" FontWeight="600" :Text="ext?.name ?? '扩展详情'" />
            </div>
            <WinButton
              Style="SubtleButtonStyle"
              aria-label="关闭详情"
              aria-hidden="true"
              @Click="onOpenChange(false)"
            >
              <span class="ppc-drawer-close-glyph">&#xE8BB;</span>
            </WinButton>
          </header>

          <div v-if="ext" class="ppc-drawer-body">
            <section class="ppc-drawer-section">
              <div class="ppc-drawer-row">
                <span class="ppc-drawer-row-icon">&#xE943;</span>
                <div>
                  <div class="ppc-drawer-label">版本</div>
                  <div class="ppc-drawer-value">{{ ext.version }}</div>
                </div>
              </div>

              <div v-if="ext.description" class="ppc-drawer-row">
                <span class="ppc-drawer-row-icon">&#xE8A5;</span>
                <div>
                  <div class="ppc-drawer-label">描述</div>
                  <div class="ppc-drawer-value">{{ ext.description }}</div>
                </div>
              </div>

              <div v-if="ext.author" class="ppc-drawer-row">
                <span class="ppc-drawer-row-icon">&#xE77B;</span>
                <div>
                  <div class="ppc-drawer-label">作者</div>
                  <div class="ppc-drawer-value">{{ ext.author }}</div>
                </div>
              </div>

              <div class="ppc-drawer-row">
                <span class="ppc-drawer-row-icon">&#xE785;</span>
                <div>
                  <div class="ppc-drawer-label">类型</div>
                  <div class="ppc-drawer-value">{{ getExtensionTypeLabel(ext) }}</div>
                </div>
              </div>

              <div v-if="ext.tags && ext.tags.length > 0" class="ppc-drawer-row">
                <span class="ppc-drawer-row-icon">&#xE8EC;</span>
                <div>
                  <div class="ppc-drawer-label">标签</div>
                  <div class="ppc-drawer-tags">
                    <span v-for="tag in ext.tags" :key="tag" class="ppc-drawer-tag">{{ tag }}</span>
                  </div>
                </div>
              </div>
            </section>

            <div class="ppc-drawer-divider" />

            <div class="ppc-drawer-switch-row">
              <div>
                <div class="ppc-drawer-switch-title">{{ ext.enabled ? '已启用' : '已禁用' }}</div>
                <div class="ppc-drawer-switch-desc">{{ ext.enabled ? '扩展正在运行' : '扩展已停止' }}</div>
              </div>
              <WinToggleSwitch
                :IsOn="ext.enabled"
                :IsEnabled="!toggling"
                :OnContent="'启用中'"
                :OffContent="'已禁用'"
                @Toggled="onToggle(ext)"
              />
            </div>

            <template v-if="configEntries.length > 0">
              <div class="ppc-drawer-divider" />
              <div class="ppc-drawer-subtitle">配置概览</div>
              <div class="ppc-drawer-file-list">
                <div v-for="[key, value] in configEntries" :key="key" class="ppc-drawer-file-item">
                  {{ key }}: {{ formatValue(value) }}
                </div>
              </div>
            </template>

            <template v-if="ext.files && ext.files.length > 0">
              <div class="ppc-drawer-divider" />
              <div class="ppc-drawer-subtitle">文件列表</div>
              <div class="ppc-drawer-file-list">
                <div v-for="file in ext.files" :key="file" class="ppc-drawer-file-item">
                  <span class="ppc-drawer-file-icon">&#xE8A5;</span>
                  <span>{{ file }}</span>
                </div>
              </div>
            </template>
          </div>
        </aside>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.ppc-drawer-root {
  position: fixed;
  inset: 0;
  z-index: 9000;
}
.ppc-drawer-backdrop {
  position: absolute;
  inset: 0;
  background: var(--dialog-overlay, rgba(0, 0, 0, 0.4));
}
.ppc-drawer-panel {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: min(100%, 420px);
  display: flex;
  flex-direction: column;
  background: var(--dialog-background, var(--app-bg));
  border-left: 1px solid var(--ctrl-border, rgba(0, 0, 0, 0.16));
  box-shadow: -16px 0 32px rgba(0, 0, 0, 0.14);
}
.ppc-drawer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 16px 20px;
  border-bottom: 1px solid var(--ctrl-border, rgba(0, 0, 0, 0.12));
}
.ppc-drawer-title-group {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
.ppc-drawer-icon {
  font-family: 'WinUIOnWebIcons', 'Segoe Fluent Icons', 'Segoe MDL2 Assets';
  font-size: 20px;
  flex-shrink: 0;
}
.ppc-drawer-close-glyph {
  font-family: 'WinUIOnWebIcons', 'Segoe Fluent Icons', 'Segoe MDL2 Assets';
  font-size: 16px;
}
.ppc-drawer-body {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  color: var(--text-primary, #000);
}
.ppc-drawer-section {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.ppc-drawer-row {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 16px;
  min-width: 0;
}
.ppc-drawer-row-icon {
  font-family: 'WinUIOnWebIcons', 'Segoe Fluent Icons', 'Segoe MDL2 Assets';
  font-size: 16px;
  color: var(--text-tertiary, #777);
  flex-shrink: 0;
  margin-top: 1px;
}
.ppc-drawer-label {
  font-size: 12px;
  color: var(--text-tertiary, #777);
  margin-bottom: 2px;
}
.ppc-drawer-value {
  font-size: 14px;
  color: var(--text-primary, #000);
  word-break: break-word;
}
.ppc-drawer-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.ppc-drawer-tag {
  display: inline-flex;
  align-items: center;
  height: 24px;
  padding: 0 10px;
  font-size: 12px;
  border-radius: 6px;
  background: var(--card-bg, rgba(0, 0, 0, 0.06));
  color: var(--text-secondary, #555);
  white-space: nowrap;
}
.ppc-drawer-divider {
  height: 1px;
  background: var(--ctrl-border, rgba(0, 0, 0, 0.12));
  margin: 18px 0;
}
.ppc-drawer-switch-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.ppc-drawer-switch-title {
  font-weight: 600;
  color: var(--text-primary, #000);
}
.ppc-drawer-switch-desc {
  font-size: 12px;
  color: var(--text-secondary, #555);
  margin-top: 2px;
}
.ppc-drawer-subtitle {
  font-weight: 600;
  color: var(--text-primary, #000);
  margin-bottom: 8px;
}
.ppc-drawer-file-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.ppc-drawer-file-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 0;
  font-size: 12px;
  color: var(--text-secondary, #555);
  font-family: monospace;
  word-break: break-all;
}
.ppc-drawer-file-icon {
  font-family: 'WinUIOnWebIcons', 'Segoe Fluent Icons', 'Segoe MDL2 Assets';
  color: var(--text-tertiary, #777);
}

.ppc-drawer-enter-active .ppc-drawer-backdrop,
.ppc-drawer-leave-active .ppc-drawer-backdrop {
  transition: opacity 250ms cubic-bezier(0.1, 0.9, 0.2, 1);
}
.ppc-drawer-enter-from .ppc-drawer-backdrop,
.ppc-drawer-leave-to .ppc-drawer-backdrop {
  opacity: 0;
}
.ppc-drawer-enter-active .ppc-drawer-panel,
.ppc-drawer-leave-active .ppc-drawer-panel {
  transition: transform 250ms cubic-bezier(0.1, 0.9, 0.2, 1);
}
.ppc-drawer-enter-from .ppc-drawer-panel,
.ppc-drawer-leave-to .ppc-drawer-panel {
  transform: translateX(100%);
}
.ppc-drawer-enter-active {
  transition: opacity 250ms cubic-bezier(0.1, 0.9, 0.2, 1);
}
.ppc-drawer-leave-active {
  transition: opacity 250ms cubic-bezier(0.1, 0.9, 0.2, 1);
}
.ppc-drawer-enter-from,
.ppc-drawer-leave-to {
  opacity: 0;
}

@media (prefers-reduced-motion: reduce) {
  .ppc-drawer-enter-active,
  .ppc-drawer-leave-active,
  .ppc-drawer-enter-active .ppc-drawer-panel,
  .ppc-drawer-leave-active .ppc-drawer-panel,
  .ppc-drawer-enter-active .ppc-drawer-backdrop,
  .ppc-drawer-leave-active .ppc-drawer-backdrop {
    transition: none;
  }
}
</style>