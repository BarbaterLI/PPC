<script setup lang="ts">
// 快捷操作区块：开始转换 + 配置管理。
// 对齐 React `src/components/dashboard/QuickActionsSection.tsx`。
import { useRouter } from 'vue-router'
import { WinButton, WinTextBlock } from 'winui-web-library'

const router = useRouter()
</script>

<template>
  <section class="ppc-quick-actions">
    <WinTextBlock class="ppc-section-title" Text="快捷操作" />
    <div class="ppc-quick-grid">
      <WinButton
        class="ppc-quick-btn ppc-quick-btn-primary"
        HorizontalContentAlignment="Stretch"
        VerticalContentAlignment="Stretch"
        @Click="router.push('/tts-convert')"
      >
        <span class="ppc-quick-inner">
          <span class="ppc-quick-icon" aria-hidden="true">&#xE768;</span>
          <span class="ppc-quick-text">开始转换</span>
        </span>
      </WinButton>
      <WinButton
        class="ppc-quick-btn"
        HorizontalContentAlignment="Stretch"
        VerticalContentAlignment="Stretch"
        @Click="router.push('/config')"
      >
        <span class="ppc-quick-inner">
          <span class="ppc-quick-icon" aria-hidden="true">&#xE713;</span>
          <span class="ppc-quick-text">配置管理</span>
        </span>
      </WinButton>
    </div>
  </section>
</template>

<style scoped>
.ppc-quick-actions {
  display: flex;
  flex-direction: column;
}
.ppc-quick-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}
@media (max-width: 500px) {
  .ppc-quick-grid {
    grid-template-columns: 1fr;
  }
}
.ppc-quick-btn {
  min-height: 96px;
  background-color: var(--layer-default, #f3f3f3);
  border: 1px solid var(--ctrl-border, rgba(0, 0, 0, 0.08));
  border-radius: 9px;
}
.ppc-quick-btn-primary {
  background-color: var(--accent-base);
  border-color: transparent;
}
.ppc-quick-inner {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
.ppc-quick-icon {
  font-family: 'WinUIOnWebIcons', 'Segoe Fluent Icons', 'Segoe MDL2 Assets';
  font-size: 34px;
  line-height: 1;
}
.ppc-quick-text {
  font-size: 14px;
  font-weight: 600;
}
.ppc-quick-btn-primary .ppc-quick-text,
.ppc-quick-btn-primary .ppc-quick-icon {
  color: #ffffff;
}
</style>