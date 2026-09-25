<script setup lang="ts">
/**
 * 扩展管理页面 —— 复刻 React 工程 `src/pages/ExtensionsPage.tsx`。
 *
 * 功能对齐：
 * - 扩展清单（useApi 拉取 `/api/extensions`）
 * - 安装扩展（.ppcext.zip 上传，带进度，WinContentDialog）
 * - 行内 启用/禁用 / 卸载（卸载经共享 ConfirmDialog 二次确认）
 * - 详情抽屉（WinUI 右侧滑出面板）
 * - 加载骨架 / 错误态（EmptyState 重试）/ 空态
 *
 * 视觉对齐：WinUI 样式（WinButton / WinProgressBar / WinProgressRing / WinToggleSwitch 等）。
 */
import { onMounted, ref } from 'vue'
import { WinButton } from 'winui-web-library'
import { PageHeader, EmptyState, LoadingSkeleton, ConfirmDialog } from '@/components'
import { useApi } from '@/composables/useApi'
import { extensionApi } from '@/api/client'
import type { Extension } from '@/api/types'
import ExtensionTable from '@/components/extensions/ExtensionTable.vue'
import ExtensionInstallDialog from '@/components/extensions/ExtensionInstallDialog.vue'
import ExtensionDetailDrawer from '@/components/extensions/ExtensionDetailDrawer.vue'
import { useExtensionActions } from '@/components/extensions/useExtensionActions'

onMounted(() => {
  document.title = '扩展管理 - PPC_NEXT'
})

// 扩展清单
const { data: extensions, loading, error, refetch } = useApi<Extension[]>(() =>
  extensionApi.list(),
)

// 动作（启停 / 卸载）
const { togglingExt, handleToggleEnabled, uninstallExtension } = useExtensionActions(refetch)

// 安装对话框
const installDialogOpen = ref(false)

// 详情抽屉
const selectedExt = ref<Extension | null>(null)
const detailDrawerOpen = ref(false)

const handleShowDetail = (ext: Extension) => {
  selectedExt.value = ext
  detailDrawerOpen.value = true
}

const openInstallDialog = () => {
  installDialogOpen.value = true
}

// 卸载二次确认（命令式 ConfirmDialog）
const confirmDialogRef = ref<InstanceType<typeof ConfirmDialog> | null>(null)

const handleUninstallClick = async (ext: Extension) => {
  if (!confirmDialogRef.value) return
  const ok = await confirmDialogRef.value.confirm({
    title: '确认卸载',
    message: `确定要卸载扩展 ${ext.name} 吗？此操作无法撤销。`,
    okText: '卸载',
    cancelText: '取消',
    danger: true,
  })
  if (ok) {
    await uninstallExtension(ext)
  }
}
</script>

<template>
  <div class="ppc-page">
    <div class="ppc-page-container">
      <PageHeader title="扩展" description="扩展插件管理">
        <template #actions>
          <WinButton Style="AccentButtonStyle" Content="+ 安装扩展" @Click="openInstallDialog" />
        </template>
      </PageHeader>

      <div class="ppc-extensions-list">
        <!-- 加载骨架 -->
        <LoadingSkeleton v-if="loading && !extensions" variant="card" :count="5" />

        <!-- 错误态 -->
        <EmptyState
          v-else-if="error"
          icon="\uE783"
          title="加载失败"
          :message="`加载扩展列表失败: ${error}`"
          action-label="重试"
          :on-action="refetch"
        />

        <!-- 空态 -->
        <EmptyState
          v-else-if="!extensions || extensions.length === 0"
          icon="\uE8B3"
          title="暂无扩展"
          message="还没有安装任何扩展插件。点击下方「安装扩展」按钮来添加新的扩展。"
          action-label="安装扩展"
          :on-action="openInstallDialog"
        />

        <!-- 扩展列表 -->
        <ExtensionTable
          v-else
          :extensions="extensions"
          :toggling-ext="togglingExt"
          :on-show-detail="handleShowDetail"
          :on-toggle="handleToggleEnabled"
          :on-uninstall-click="handleUninstallClick"
        />
      </div>

      <ExtensionInstallDialog
        :open="installDialogOpen"
        :on-open-change="(v: boolean) => (installDialogOpen = v)"
        :on-installed="refetch"
      />

      <ExtensionDetailDrawer
        :open="detailDrawerOpen"
        :ext="selectedExt"
        :toggling="selectedExt !== null && togglingExt === selectedExt.name"
        :on-open-change="(v: boolean) => (detailDrawerOpen = v)"
        :on-toggle="handleToggleEnabled"
      />

      <ConfirmDialog ref="confirmDialogRef" />
    </div>
  </div>
</template>

<style scoped>
.ppc-extensions-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
</style>