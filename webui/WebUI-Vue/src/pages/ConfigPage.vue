<script setup lang="ts">
// 配置管理页 —— 从 React `src/pages/ConfigPage.tsx` 迁移。
// 支持配置分类分组、按类型编辑、批量保存、导入/导出、重置确认、分步向导。
import { computed, onMounted, ref, watch } from 'vue'
import { WinButton } from 'winui-web-library'
import { PageHeader, EmptyState, LoadingSkeleton, ConfirmDialog } from '@/components'
import ConfigField from '@/components/config/ConfigField.vue'
import ConfigWizard from '@/components/config/ConfigWizard.vue'
import { useApi } from '@/composables/useApi'
import { useToast } from '@/composables/useToast'
import { configApi } from '@/api/client'
import type { PPCConfig } from '@/api/types'

const CATEGORY_TABS: Record<string, string> = {
  core: '核心',
  tts: 'TTS',
  split: '分割',
  performance: '性能',
  network: '网络',
  reliability: '可靠性',
  extension: '扩展',
  output: '输出',
}

const TAB_ORDER = ['core', 'tts', 'split', 'performance', 'network', 'reliability', 'extension', 'output']

type ConfigMap = Record<string, string>

const { showToast } = useToast()

onMounted(() => {
  document.title = '配置管理 - PPC_NEXT'
})

const { data: configs, loading, refetch } = useApi<PPCConfig[]>(() => configApi.getAll())
const confirmDialog = ref<InstanceType<typeof ConfirmDialog> | null>(null)
const fileInputRef = ref<HTMLInputElement | null>(null)

const editedValues = ref<ConfigMap>({})
const saving = ref(false)
const exporting = ref(false)
const importing = ref(false)
const resetting = ref(false)
const wizardOpen = ref(false)
const selectedTab = ref<string>('')
try {
  selectedTab.value = localStorage.getItem('config-selected-tab') || ''
} catch {
  /* storage unavailable; ignore */
}

watch(selectedTab, (v) => {
  try {
    localStorage.setItem('config-selected-tab', v)
  } catch {
    /* storage unavailable; silently ignore */
  }
})

const groupedConfigs = computed<Record<string, PPCConfig[]>>(() => {
  const groups: Record<string, PPCConfig[]> = {}
  for (const cfg of configs.value ?? []) {
    const cat = cfg.category || 'core'
    if (!groups[cat]) groups[cat] = []
    groups[cat].push(cfg)
  }
  return groups
})

const originalValues = computed<ConfigMap>(() => {
  const map: ConfigMap = {}
  for (const cfg of configs.value ?? []) {
    map[cfg.key] = cfg.value ?? cfg.default_value ?? ''
  }
  return map
})

const availableTabs = computed(() => TAB_ORDER.filter((cat) => (groupedConfigs.value[cat]?.length ?? 0) > 0))

const activeTab = computed(() => {
  if (availableTabs.value.includes(selectedTab.value)) return selectedTab.value
  return availableTabs.value[0] || 'core'
})

const activeTabItems = computed(() => groupedConfigs.value[activeTab.value] ?? [])

const changedCount = computed(
  () =>
    Object.entries(editedValues.value).filter(
      ([key, value]) => value !== (originalValues.value[key] ?? ''),
    ).length,
)

const getValue = (cfg: PPCConfig): string => {
  if (cfg.key in editedValues.value) return editedValues.value[cfg.key] ?? ''
  return cfg.value ?? cfg.default_value ?? ''
}

const isChanged = (cfg: PPCConfig): boolean => {
  if (!(cfg.key in editedValues.value)) return false
  const original = cfg.value ?? cfg.default_value ?? ''
  return editedValues.value[cfg.key] !== original
}

const handleValueChange = (key: string, value: string) => {
  editedValues.value = { ...editedValues.value, [key]: value }
}

const handleSave = async () => {
  if (!configs.value || changedCount.value === 0) return
  saving.value = true
  try {
    const payloads: Array<{ key: string; value: string }> = []
    for (const [key, value] of Object.entries(editedValues.value)) {
      const cfg = configs.value.find((c) => c.key === key)
      if (!cfg) continue
      const original = cfg.value ?? cfg.default_value ?? ''
      if (value === original) continue
      payloads.push({ key, value })
    }

    if (payloads.length === 0) {
      showToast({ title: '没有需要保存的更改', intent: 'info' })
      return
    }

    const result = await configApi.batchUpdate(payloads)
    if (result.success === false) {
      throw new Error(result.error?.message || '批量保存失败')
    }
    const batchData = result.data
    if (batchData?.failed && batchData.failed.length > 0) {
      const failedKeys = batchData.failed.map((f) => f.key).join(', ')
      showToast({ title: '部分保存失败', body: `失败项: ${failedKeys}`, intent: 'warning' })
    } else {
      showToast({ title: '配置已保存', body: `成功保存 ${payloads.length} 项配置`, intent: 'success' })
    }
    editedValues.value = {}
    refetch()
  } catch (err: unknown) {
    showToast({ title: '保存失败', body: err instanceof Error ? err.message : '保存失败', intent: 'error' })
  } finally {
    saving.value = false
  }
}

const handleExport = async () => {
  exporting.value = true
  try {
    const blob = await configApi.exportFile()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ppc-config-${new Date().toISOString().slice(0, 10)}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    showToast({ title: '导出成功', body: '配置文件已下载', intent: 'success' })
  } catch (err: unknown) {
    showToast({ title: '导出失败', body: err instanceof Error ? err.message : '导出失败', intent: 'error' })
  } finally {
    exporting.value = false
  }
}

const handleImportClick = () => {
  fileInputRef.value?.click()
}

const handleImportFile = async (e: Event) => {
  const target = e.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return
  importing.value = true
  try {
    await configApi.importFile(file)
    showToast({ title: '导入成功', body: `已从 ${file.name} 导入配置`, intent: 'success' })
    editedValues.value = {}
    refetch()
  } catch (err: unknown) {
    showToast({ title: '导入失败', body: err instanceof Error ? err.message : '导入失败', intent: 'error' })
  } finally {
    importing.value = false
    target.value = ''
  }
}

const handleResetClick = async () => {
  const ok = await confirmDialog.value?.confirm({
    title: '重置配置',
    message: '确定要将所有配置重置为默认值吗？此操作不可撤销。',
    okText: '确认重置',
    cancelText: '取消',
    danger: true,
  })
  if (!ok) return
  resetting.value = true
  try {
    const result = await configApi.reset()
    if (result.success === false) {
      throw new Error(result.error?.message || '重置失败')
    }
    showToast({ title: '重置成功', body: '配置已重置为默认值', intent: 'success' })
    editedValues.value = {}
    refetch()
  } catch (err: unknown) {
    showToast({ title: '重置失败', body: err instanceof Error ? err.message : '重置失败', intent: 'error' })
  } finally {
    resetting.value = false
  }
}

const handleWizardComplete = (data: {
  voice: string
  concurrency: number
  rate: string
  maxRetries: number
  outputFormat: string
  outputBackend: string
}) => {
  editedValues.value = {
    ...editedValues.value,
    'tts.default_voice': data.voice,
    'tts.concurrency': String(data.concurrency),
    'tts.rate': data.rate,
    'tts.retries': String(data.maxRetries),
    'tts.backend': data.outputBackend,
    'output.default_format': data.outputFormat,
  }
  wizardOpen.value = false
  showToast({ title: '向导完成', body: '配置已更新，请点击保存以应用更改', intent: 'info' })
}

const hasChanges = computed(() => changedCount.value > 0)
</script>

<template>
  <div class="ppc-page">
    <div class="ppc-page-container ppc-page-container--narrow">
      <PageHeader title="配置" description="管理系统核心配置参数">
        <template #actions>
          <WinButton
            Style="AccentButtonStyle"
            :IsEnabled="hasChanges && !saving"
            @Click="handleSave"
          >
            <span class="ppc-btn-inner">
              <span class="ppc-icon">&#xE74E;</span>保存
              <span v-if="hasChanges" class="ppc-badge">{{ changedCount }}</span>
            </span>
          </WinButton>
        </template>
      </PageHeader>

      <LoadingSkeleton v-if="loading && !configs" :count="6" variant="card" />

      <EmptyState
        v-else-if="configs && configs.length === 0"
        title="暂无配置"
        message="当前没有可用的配置参数。"
      />

      <div v-else-if="configs" class="ppc-content">
      <div class="ppc-toolbar">
        <WinButton Style="SubtleButtonStyle" :IsEnabled="!importing" @Click="handleImportClick">
          <span class="ppc-btn-inner"><span class="ppc-icon">&#xE898;</span>导入</span>
        </WinButton>
        <WinButton Style="SubtleButtonStyle" :IsEnabled="!exporting" @Click="handleExport">
          <span class="ppc-btn-inner"><span class="ppc-icon">&#xE896;</span>导出</span>
        </WinButton>
        <WinButton Style="SubtleButtonStyle" :IsEnabled="!resetting" @Click="handleResetClick">
          <span class="ppc-btn-inner"><span class="ppc-icon">&#xE72C;</span>重置</span>
        </WinButton>
        <WinButton Style="SubtleButtonStyle" @Click="wizardOpen = true">
          <span class="ppc-btn-inner"><span class="ppc-icon">&#xE713;</span>配置向导</span>
        </WinButton>
        <input
          ref="fileInputRef"
          type="file"
          accept=".json"
          class="ppc-hidden-input"
          @change="handleImportFile"
        />
      </div>

      <div v-if="availableTabs.length > 1" class="ppc-tabs" role="tablist">
        <button
          v-for="cat in availableTabs"
          :key="cat"
          type="button"
          role="tab"
          class="ppc-tab"
          :class="{ 'ppc-tab--active': cat === activeTab }"
          @click="selectedTab = cat"
        >
          {{ CATEGORY_TABS[cat] ?? cat }}
        </button>
      </div>

      <div class="ppc-tab-content">
        <template v-if="activeTabItems.length > 0">
          <ConfigField
            v-for="cfg in activeTabItems"
            :key="cfg.key"
            :config="cfg"
            :value="getValue(cfg)"
            :isChanged="isChanged(cfg)"
            :onChange="(v) => handleValueChange(cfg.key, v)"
          />
        </template>
        <EmptyState v-else title="此分类下没有配置项" message="当前分类没有可用的配置参数。" />
      </div>

      <div class="ppc-save-bar">
        <WinButton
          Style="AccentButtonStyle"
          :IsEnabled="hasChanges && !saving"
          @Click="handleSave"
        >
          <span class="ppc-btn-inner">
            <span class="ppc-icon">&#xE74E;</span>保存所有更改
            <span v-if="hasChanges" class="ppc-badge">{{ changedCount }}</span>
          </span>
        </WinButton>
      </div>
      </div>
    </div>

    <ConfirmDialog ref="confirmDialog" />
    <ConfigWizard :open="wizardOpen" @close="wizardOpen = false" @complete="handleWizardComplete" />
  </div>
</template>

<style scoped>
.ppc-content {
  display: flex;
  flex-direction: column;
}
.ppc-toolbar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 16px 0;
}
.ppc-hidden-input {
  display: none;
}
.ppc-tabs {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px;
  border-bottom: 1px solid var(--ctrl-border-accent, rgba(0, 0, 0, 0.16));
  margin-bottom: 16px;
}
.ppc-tab {
  appearance: none;
  border: 0;
  background: transparent;
  color: var(--text-secondary, #555);
  font-size: 14px;
  padding: 8px 14px;
  cursor: pointer;
  border-radius: 4px;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
}
.ppc-tab:hover {
  background: var(--subtle-fill-color-secondary, #eee);
}
.ppc-tab--active {
  color: var(--accent-base, #0067c0);
  font-weight: 600;
  border-bottom-color: var(--accent-base, #0067c0);
}
.ppc-tab-content {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.ppc-save-bar {
  margin-top: 24px;
}
.ppc-btn-inner {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.ppc-icon {
  font-family: 'WinUIOnWebIcons', 'Segoe Fluent Icons', 'Segoe MDL2 Assets';
  font-size: 14px;
  line-height: 1;
}
.ppc-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 18px;
  height: 18px;
  padding: 0 5px;
  border-radius: 9px;
  background: #f7b84b;
  color: #1a1a1a;
  font-size: 11px;
  font-weight: 700;
  line-height: 1;
}
</style>