<script setup lang="ts">
// 应用根组件：全局错误边界 + 全局 Toast 容器 + 路由出口。
import ErrorBoundary from '@/components/ErrorBoundary.vue'
import ToastProvider from '@/components/ToastProvider.vue'
</script>

<template>
  <ErrorBoundary>
    <ToastProvider>
      <router-view />
    </ToastProvider>
  </ErrorBoundary>
</template>