<script setup lang="ts">
// 配置重置对话框：选择预设方案后确认重置。
// 对齐 React `src/components/config/ConfigResetDialog.tsx`。
import { ref } from 'vue'
import { WinContentDialog, WinTextBlock, WinComboBox } from 'winui-web-library'

const PRESET_OPTIONS = [
  { key: 'balanced', label: '均衡模式' },
  { key: 'speed', label: '速度优先' },
  { key: 'quality', label: '质量优先' },
]

defineProps<{ open: boolean; resetting: boolean }>()
const emit = defineEmits<{
  closeGoogle: []
  reset: [preset: string]
}>()

const preset = ref('balanced')

const handleReset = () => {
  emit('reset', preset.value)
}
</script>

<template>
  <WinContentDialog
    :IsOpen="open"
    Title="重置配置"
    DefaultButton="Close"
    :PrimaryButtonText="'取消'"
    :SecondaryButtonText="'确认重置'"
    IsPrimaryButtonEnabled
    :IsSecondaryButtonEnabled="!resetting && false"
    @PrimaryButtonClick="emit('closeGoogle')"
    @SecondaryButtonClick="handleReset"
    @CloseButtonClick="emit('closeGoogle')"
  >
    <div class="ppc-reset-content">
      <WinTextBlock Text="选择预设配置方案以重置所有设置：" />
      <WinComboBox
        :ItemsSource="PRESET_OPTIONS"
        DisplayMemberPath="label"
        SelectedValuePath="key"
        :SelectedValue="preset"
        :MinWidth="200"
        @update:SelectedValue="(v: unknown) => (preset = v == null ? 'balanced' : String(v))"
      />
    </div>
  </WinContentDialog>
</template>

<style scoped>
.ppc-reset-content {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
</style>