<script setup lang="ts">
// 错误边界（Vue 实现）：捕获子组件渲染/生命周期错误，提供统一错误页 + 重试 + 返回。
import { onErrorCaptured, ref } from 'vue'
import { useRouter } from 'vue-router'
import { WinButton, WinTextBlock } from 'winui-web-library'

const hasError = ref(false)
const error = ref<Error | null>(null)
const router = useRouter()

onErrorCaptured((err) => {
  hasError.value = true
  error.value = err instanceof Error ? err : new Error(String(err))
  // 记录但不向上冒泡
  return false
})

const reset = () => {
  hasError.value = false
  error.value = null
}

const goHome = () => {
  reset()
  router.push('/')
}
</script>

<template>
  <slot v-if="!hasError" />
  <div v-else class="ppc-error" role="alert">
    <span class="ppc-error-icon" aria-hidden="true">&#xE8F1;</span>
    <WinTextBlock class="ppc-error-title" FontSize="20" FontWeight="600" Text="出现错误" />
    <WinTextBlock
      class="ppc-error-message"
      FontSize="14"
      Text="应用遇到了意外错误。请尝试刷新页面或联系管理员。"
    />
    <div class="ppc-error-actions">
      <WinButton Style="AccentButtonStyle" Content="重试" @Click="reset" />
      <WinButton Style="SubtleButtonStyle" Content="返回首页" @Click="goHome" />
    </div>
    <pre v-if="error" class="ppc-error-details">{{ error.name }}: {{ error.message }}</pre>
  </div>
</template>

<style scoped>
.ppc-error {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 300px;
  padding: 48px 32px;
  text-align: center;
  color: var(--text-primary, #000);
}
.ppc-error-icon {
  font-family: 'WinUIOnWebIcons', 'Segoe Fluent Icons', 'Segoe MDL2 Assets';
  font-size: 48px;
  color: #c42b1c;
  line-height: 1;
}
.ppc-error-title {
  color: var(--text-primary, #000);
}
.ppc-error-message {
  color: var(--text-secondary, #555);
  max-width: 460px;
}
.ppc-error-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
  flex-wrap: wrap;
  justify-content: center;
}
.ppc-error-details {
  max-width: 480px;
  margin-top: 12px;
  padding: 12px;
  background: var(--card-bg, #f3f3f3);
  border-radius: 6px;
  font-family: Consolas, 'Courier New', monospace;
  font-size: 12px;
  color: var(--text-tertiary, #777);
  white-space: pre-wrap;
  word-break: break-all;
  text-align: left;
  max-height: 200px;
  overflow: auto;
}
</style>