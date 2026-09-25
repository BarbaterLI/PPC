<script setup lang="ts">
// 扩展 WebUI 页 —— 动态加载扩展包内 webui/ 页面。
// 路由 /ext/:name，依据 /api/extensions/webui 返回的配置决定加载方式：
// - mode=iframe（默认）：iframe 内嵌扩展包静态页（如 fanqie 的 /extensions/fanqie）
// - mode=component：未来可映射到内置组件（component_name）
import { computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { WinButton, WinTextBlock } from 'winui-web-library'
import { EmptyState, LoadingSkeleton, PageHeader } from '@/components'
import { useApi } from '@/composables/useApi'
import { extensionApi } from '@/api/client'
import type { ExtensionWebUIConfig } from '@/api/types'

const props = defineProps<{ name: string }>()

const router = useRouter()

const extName = computed(() => String(props.name ?? ''))

const { data: configs, loading, error, refetch } = useApi<ExtensionWebUIConfig[]>(() =>
  extensionApi.getWebUIConfigs(),
)

const config = computed(() => configs.value?.find((c) => c.extension_name === extName.value))

const notFound = computed(
  () => !loading.value && !error.value && !!configs.value && !config.value,
)

const iframeSrc = computed(() => {
  const route = config.value?.route
  if (typeof route === 'string' && route) return route
  return `/extensions/${encodeURIComponent(extName.value)}/`
})

// component 模式由内置组件渲染；其余（含无配置的容错路径）一律 iframe。
const isIframe = computed(() => !config.value || (config.value.mode ?? 'iframe') === 'iframe')

onMounted(applyTitle)

watch([config, extName], applyTitle)

function applyTitle() {
  const t = config.value?.title
  document.title = t ? `${t} - PPC_NEXT` : '扩展 - PPC_NEXT'
}

const goExtensions = () => {
  void router.push('/extensions')
}

const openInNewWindow = () => {
  window.open(iframeSrc.value, '_blank', 'noopener')
}
</script>

<template>
  <div class="ppc-page">
    <div class="ppc-page-container">
      <LoadingSkeleton v-if="loading && !configs" variant="card" :count="2" />

      <EmptyState
        v-else-if="error"
        icon="\uE783"
        title="加载失败"
        :message="`加载扩展 WebUI 配置失败: ${error}`"
        action-label="重试"
        :on-action="refetch"
      />

      <EmptyState
        v-else-if="notFound"
        icon="\uE71D"
        title="未找到扩展页面"
        message="该扩展未启用或未提供 WebUI 页面，请到扩展管理中检查。"
        action-label="前往扩展管理"
        :on-action="goExtensions"
      />

      <template v-else-if="config">
        <PageHeader :title="config.title" :description="config.description" />

        <div class="ppc-ext-page-toolbar">
          <WinTextBlock class="ppc-ext-page-route" FontSize="12">
            {{ config.route }}
          </WinTextBlock>
          <WinButton
            v-if="isIframe"
            Style="SubtleButtonStyle"
            Content="在新窗口打开"
            @Click="openInNewWindow"
          />
        </div>

        <div v-if="isIframe" class="ppc-ext-iframe-container">
          <iframe
            class="ppc-ext-iframe"
            :src="iframeSrc"
            :title="config.title"
            sandbox="allow-scripts allow-forms allow-popups"
          />
        </div>
        <div v-else class="ppc-ext-component-error">
          <WinTextBlock FontSize="13">
            扩展 {{ config.extension_name }} 声明的组件模式暂不支持（component_name:
            {{ config.component_name || '-' }}）。
          </WinTextBlock>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.ppc-ext-page-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
.ppc-ext-page-route {
  font-family: Consolas, monospace;
  color: var(--text-secondary, #555);
  padding: 4px 8px;
  border-radius: 4px;
  background-color: var(--layer-default, #f3f3f3);
}
.ppc-ext-iframe-container {
  width: 100%;
  height: min(70vh, 700px);
  min-height: 480px;
  border: 1px solid var(--ctrl-border, rgba(0, 0, 0, 0.16));
  border-radius: 9px;
  overflow: hidden;
  background-color: var(--layer-default, #fff);
}
.ppc-ext-iframe {
  width: 100%;
  height: 100%;
  border: none;
}
.ppc-ext-component-error {
  padding: 24px;
  border: 1px dashed var(--ctrl-border, rgba(0, 0, 0, 0.16));
  border-radius: 9px;
  color: var(--text-secondary, #555);
}
</style>