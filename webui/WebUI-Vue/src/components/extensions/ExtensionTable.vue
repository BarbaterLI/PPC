<script setup lang="ts">
// 扩展表格 —— 复刻 React 工程 `src/components/extensions/ExtensionTable.tsx`。
import { computed } from 'vue'
import { WinButton, WinProgressRing, WinToolTip } from 'winui-web-library'
import { StatusBadge } from '@/components'
import { designTokens } from '@/theme/tokens'
import { getExtensionTypeLabel } from './extensionType'
import type { Extension } from '@/api/types'

defineProps<{
  extensions: Extension[]
  togglingExt: string | null
  onShowDetail: (ext: Extension) => void
  onToggle: (ext: Extension) => void
  onUninstallClick: (ext: Extension) => void
}>()

const accentColor = computed(() => designTokens.emphasis.accent)
</script>

<template>
  <div class="ppc-extension-table-wrapper">
    <table class="ppc-extension-table">
      <thead>
        <tr>
          <th>名称</th>
          <th>版本</th>
          <th>类型</th>
          <th>状态</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="ext in extensions" :key="ext.id ?? ext.name" class="ppc-extension-row">
          <td>
            <div class="ppc-extension-cell-name">
              <span class="ppc-extension-icon" :style="{ color: accentColor }">
                {{ ext.has_webui ? '\uE774' : '\uE8B3' }}
              </span>
              <div class="ppc-extension-cell-text">
                <div class="ppc-extension-name">{{ ext.name }}</div>
                <div v-if="ext.description" class="ppc-extension-desc">{{ ext.description }}</div>
              </div>
            </div>
          </td>
          <td>
            <span class="ppc-extension-version">{{ ext.version }}</span>
          </td>
          <td>
            <span class="ppc-extension-type">{{ getExtensionTypeLabel(ext) }}</span>
          </td>
          <td>
            <WinProgressRing
              v-if="togglingExt === ext.name"
              :IsActive="true"
              :IsIndeterminate="true"
              :Width="16"
              :Height="16"
            />
            <StatusBadge
              v-else
              :status="ext.enabled ? 'completed' : 'cancelled'"
              :label="ext.enabled ? '已启用' : '已禁用'"
            />
          </td>
          <td>
            <div class="ppc-extension-actions">
              <WinToolTip :Content="'查看详情'">
                <WinButton Style="SubtleButtonStyle" :Content="'详情'" @Click="onShowDetail(ext)" />
              </WinToolTip>
              <WinToolTip :Content="ext.enabled ? '禁用' : '启用'">
                <WinButton
                  Style="SubtleButtonStyle"
                  :Content="ext.enabled ? '禁用' : '启用'"
                  :IsEnabled="togglingExt !== ext.name"
                  @Click="onToggle(ext)"
                />
              </WinToolTip>
              <WinToolTip :Content="'卸载扩展'">
                <WinButton Style="SubtleButtonStyle" :Content="'卸载'" @Click="onUninstallClick(ext)" />
              </WinToolTip>
            </div>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.ppc-extension-table-wrapper {
  overflow-x: auto;
  background-color: var(--layer-default);
  border: 1px solid var(--ctrl-border);
  border-radius: 9px;
}
.ppc-extension-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
.ppc-extension-table th {
  text-align: left;
  padding: 12px 16px;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary, #555);
  border-bottom: 1px solid var(--ctrl-border, rgba(0, 0, 0, 0.12));
  background-color: var(--card-bg, transparent);
  white-space: nowrap;
}
.ppc-extension-table td {
  padding: 12px 16px;
  color: var(--text-primary, #000);
  border-bottom: 1px solid var(--ctrl-border, rgba(0, 0, 0, 0.12));
  vertical-align: middle;
}
.ppc-extension-row:last-child td {
  border-bottom: none;
}
.ppc-extension-row:hover {
  background-color: var(--hover-bg, rgba(0, 0, 0, 0.04));
}
.ppc-extension-cell-name {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
.ppc-extension-icon {
  font-family: 'WinUIOnWebIcons', 'Segoe Fluent Icons', 'Segoe MDL2 Assets';
  font-size: 18px;
  flex-shrink: 0;
}
.ppc-extension-cell-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.ppc-extension-name {
  font-weight: 600;
  color: var(--text-primary, #000);
}
.ppc-extension-desc {
  font-size: 12px;
  color: var(--text-secondary, #555);
  max-width: 40ch;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ppc-extension-version {
  font-family: monospace;
  font-size: 12px;
  color: var(--text-secondary, #555);
}
.ppc-extension-type {
  font-size: 12px;
  color: var(--text-secondary, #555);
}
.ppc-extension-actions {
  display: flex;
  align-items: center;
  gap: 4px;
  white-space: nowrap;
}

@media (max-width: 767px) {
  .ppc-extension-desc {
    max-width: 24ch;
  }
}
</style>