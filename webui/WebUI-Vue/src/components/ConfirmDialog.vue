<script setup lang="ts">
// 确认对话框：复用 WinContentDialog，支持命令式（defineExpose.confirm）与模板插槽两种用法。
import { ref } from 'vue'
import { WinContentDialog } from 'winui-web-library'

interface ConfirmOptions {
  title?: string
  message: string
  okText?: string
  cancelText?: string
  danger?: boolean
}

const isOpen = ref(false)
const options = ref<ConfirmOptions>({ message: '' })
let resolver: ((value: boolean) => void) | null = null

function confirm(message: string): Promise<boolean>
function confirm(options: ConfirmOptions): Promise<boolean>
function confirm(input: string | ConfirmOptions): Promise<boolean> {
  options.value = typeof input === 'string' ? { message: input } : input
  isOpen.value = true
  return new Promise<boolean>((resolve) => {
    resolver = resolve
  })
}

const closeWith = (value: boolean) => {
  isOpen.value = false
  resolver?.(value)
  resolver = null
}

defineExpose({ confirm })
</script>

<template>
  <WinContentDialog
    :IsOpen="isOpen"
    :Title="options.title ?? '确认操作'"
    :Content="options.message"
    :PrimaryButtonText="options.okText ?? (options.danger ? '删除' : '确认')"
    :SecondaryButtonText="options.cancelText ?? '取消'"
    :DefaultButton="options.danger ? 'Primary' : 'Secondary'"
    @PrimaryButtonClick="closeWith(true)"
    @SecondaryButtonClick="closeWith(false)"
    @CloseButtonClick="closeWith(false)"
  />
</template>