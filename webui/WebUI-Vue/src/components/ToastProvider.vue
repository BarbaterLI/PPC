<script setup lang="ts">
// 全局 Toast 容器：provide toastKey，子组件通过 useToast 注入。
import { provide, ref } from 'vue'
import { WinButton, WinInfoBar } from 'winui-web-library'
import { toastKey } from '@/composables/useToast'
import type { ToastOptions, ToastIntentType, ToastContextValue } from '@/composables/useToast'

interface ToastItem extends ToastOptions {
  id: number
}

let nextId = 0
const items = ref<ToastItem[]>([])

function toSeverity(intent?: ToastIntentType): 'Informational' | 'Success' | 'Warning' | 'Error' {
  if (intent === 'success') return 'Success'
  if (intent === 'error') return 'Error'
  if (intent === 'warning') return 'Warning'
  return 'Informational'
}

function remove(id: number) {
  items.value = items.value.filter((i) => i.id !== id)
}

function showToast(options: ToastOptions) {
  const id = ++nextId
  const base = options.intent === 'error' ? 8000 : 5000
  const duration = options.duration ?? base
  items.value.push({ ...options, id })
  if (duration > 0) {
    // 自动消失
    window.setTimeout(() => remove(id), duration)
  }
}

const context: ToastContextValue = { showToast }
provide(toastKey, context)
</script>

<template>
  <slot />
  <div class="ppc-toast-container" aria-live="polite">
    <TransitionGroup name="ppc-toast">
      <WinInfoBar
        v-for="item in items"
        :key="item.id"
        class="ppc-toast-item"
        :Severity="toSeverity(item.intent)"
        :Title="item.title"
        :Message="item.body"
        IsClosable
        @CloseButtonClick="remove(item.id)"
      >
        <template v-if="item.action" #ActionButton>
          <WinButton
            Style="AccentButtonStyle"
            :Content="item.action.label"
            @Click="
              () => {
                item.action?.onClick()
                remove(item.id)
              }
            "
          />
        </template>
      </WinInfoBar>
    </TransitionGroup>
  </div>
</template>

<style scoped>
.ppc-toast-container {
  position: fixed;
  top: 16px;
  right: 16px;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-width: min(420px, calc(100vw - 32px));
}
.ppc-toast-item {
  width: 100%;
}

.ppc-toast-enter-active,
.ppc-toast-leave-active {
  transition:
    opacity 0.2s var(--fast-out-slow-in, ease),
    transform 0.2s var(--fast-out-slow-in, ease);
}
.ppc-toast-enter-from {
  opacity: 0;
  transform: translateY(-8px);
}
.ppc-toast-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}
</style>