<script setup lang="ts">
// 安装扩展对话框 —— 复刻 React 工程 `src/components/extensions/ExtensionInstallDialog.tsx`。
// 通过拖拽/点击选择 `.ppcext.zip` 包上传安装（后端仅支持文件安装）。
import { computed, ref } from 'vue'
import { WinContentDialog, WinProgressBar, WinTextBlock } from 'winui-web-library'
import { useToast } from '@/composables/useToast'
import { extensionApi } from '@/api/client'
import { designTokens } from '@/theme/tokens'

const props = defineProps<{
  open: boolean
  onOpenChange: (open: boolean) => void
  onInstalled: () => void
}>()

const { showToast } = useToast()

const uploading = ref(false)
const uploadProgress = ref(0)
const dragOver = ref(false)

const fileInputRef = ref<HTMLInputElement | null>(null)
let abortController: AbortController | null = null

const dropBorder = computed(() => dragOver.value ? designTokens.emphasis.accent : designTokens.surface.stroke2)

const handleFileSelect = (file: File) => {
  if (!file.name.endsWith('.ppcext.zip')) {
    showToast({
      title: '文件格式错误',
      body: '请选择 .ppcext.zip 格式的扩展文件',
      intent: 'warning',
    })
    return
  }

  uploading.value = true
  uploadProgress.value = 0

  const controller = new AbortController()
  abortController = controller

  extensionApi
    .installWithProgress(
      file,
      (percent) => {
        uploadProgress.value = percent
      },
      controller.signal,
    )
    .then(() => {
      uploading.value = false
      uploadProgress.value = 0
      props.onOpenChange(false)
      showToast({
        title: '安装成功',
        body: `扩展 "${file.name}" 已安装`,
        intent: 'success',
      })
      props.onInstalled()
    })
    .catch((err: unknown) => {
      uploading.value = false
      uploadProgress.value = 0
      if (err instanceof DOMException && err.name === 'AbortError') return
      showToast({
        title: '安装失败',
        body: err instanceof Error ? err.message : '安装失败',
        intent: 'error',
      })
    })
}

const handleDrop = (e: DragEvent) => {
  e.preventDefault()
  dragOver.value = false
  const file = e.dataTransfer?.files[0]
  if (file) {
    handleFileSelect(file)
  }
}

const handleDragOver = (e: DragEvent) => {
  e.preventDefault()
  dragOver.value = true
}

const handleDragLeave = (e: DragEvent) => {
  e.preventDefault()
  dragOver.value = false
}

const handleInputChange = (e: Event) => {
  const target = e.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    handleFileSelect(file)
  }
  target.value = ''
}

const handleClose = (open: boolean) => {
  if (open) return
  abortController?.abort()
  abortController = null
  if (!uploading.value) {
    props.onOpenChange(false)
  }
}
</script>

<template>
  <WinContentDialog
    :IsOpen="open"
    :Title="'安装扩展'"
    :IsLightDismissEnabled="false"
    :SecondaryButtonText="'取消'"
    :IsSecondaryButtonEnabled="!uploading"
    @update:IsOpen="handleClose"
    @SecondaryButtonClick="handleClose(false)"
  >
    <input
      ref="fileInputRef"
      type="file"
      accept=".ppcext.zip"
      class="ppc-extension-hidden-input"
      @change="handleInputChange"
    />
    <div
      class="ppc-upload-area"
      :class="{ 'is-drag-over': dragOver }"
      :style="{ borderColor: dropBorder }"
      @click="fileInputRef?.click()"
      @drop.prevent="handleDrop"
      @dragover.prevent="handleDragOver"
      @dragleave.prevent="handleDragLeave"
    >
      <template v-if="uploading">
        <div class="ppc-upload-progress">
          <WinTextBlock class="ppc-upload-label" :Text="'正在上传...'" />
          <WinProgressBar
            :Value="uploadProgress"
            :Maximum="100"
            :Width="'100%'"
            :Height="4"
          />
          <WinTextBlock class="ppc-upload-percent" FontSize="12" :Text="`${Math.round(uploadProgress)}%`" />
        </div>
      </template>
      <template v-else>
        <span class="ppc-upload-icon" aria-hidden="true">&#xE8B5;</span>
        <WinTextBlock class="ppc-upload-text" :Text="'拖拽文件到此处，或点击选择文件'" />
        <WinTextBlock class="ppc-upload-hint" FontSize="12" :Text="'仅支持 .ppcext.zip 格式'" />
      </template>
    </div>
  </WinContentDialog>
</template>

<style scoped>
.ppc-extension-hidden-input {
  display: none;
}
.ppc-upload-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 32px;
  border: 2px dashed var(--ctrl-border-accent, rgba(0, 0, 0, 0.16));
  border-radius: 12px;
  background-color: var(--card-bg, rgba(0, 0, 0, 0.04));
  cursor: pointer;
  transition:
    border-color 250ms cubic-bezier(0.33, 0, 0.67, 1),
    background-color 250ms cubic-bezier(0.33, 0, 0.67, 1);
  text-align: center;
  gap: 8px;
}
.ppc-upload-area:hover,
.ppc-upload-area.is-drag-over {
  background-color: var(--hover-bg, rgba(0, 0, 0, 0.07));
}
.ppc-upload-icon {
  font-family: 'WinUIOnWebIcons', 'Segoe Fluent Icons', 'Segoe MDL2 Assets';
  font-size: 46px;
  line-height: 1;
  color: var(--text-tertiary, #777);
  margin-bottom: 8px;
}
.ppc-upload-text {
  color: var(--text-secondary, #555);
}
.ppc-upload-hint {
  color: var(--text-tertiary, #777);
  margin-top: 2px;
}
.ppc-upload-progress {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.ppc-upload-label {
  color: var(--text-primary, #000);
}
.ppc-upload-percent {
  color: var(--text-secondary, #555);
}
</style>