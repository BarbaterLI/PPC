<script setup lang="ts">
// 空态组件：图标 + 标题 + 描述 + 操作按钮。
import { WinButton, WinTextBlock } from 'winui-web-library'

defineProps<{
  icon?: string
  title?: string
  message: string
  actionLabel?: string
  onAction?: () => void
  secondaryActionLabel?: string
  onSecondaryAction?: () => void
}>()

const defaultIcon = '\uE8B3'
</script>

<template>
  <div class="ppc-empty">
    <WinTextBlock
      class="ppc-empty-icon"
      FontFamily="WinUIOnWebIcons"
      :Text="icon ?? defaultIcon"
      FontSize="44"
    />
    <WinTextBlock
      v-if="title"
      class="ppc-empty-title"
      FontSize="18"
      FontWeight="600"
      :Text="title"
    />
    <WinTextBlock class="ppc-empty-message" :Text="message" />
    <div v-if="$slots.actions || actionLabel" class="ppc-empty-actions">
      <slot name="actions">
        <template v-if="actionLabel && onAction">
          <WinButton Style="AccentButtonStyle" :Content="actionLabel" @Click="onAction" />
        </template>
        <WinButton
          v-if="secondaryActionLabel && onSecondaryAction"
          Style="SubtleButtonStyle"
          :Content="secondaryActionLabel"
          @Click="onSecondaryAction"
        />
      </slot>
    </div>
  </div>
</template>

<style scoped>
.ppc-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 40px 24px;
  text-align: center;
  min-height: 240px;
  border-radius: 9px;
}
.ppc-empty-icon {
  color: var(--accent-base, #53afab);
  font-size: 44px;
  line-height: 1;
  font-family: 'WinUIOnWebIcons', 'Segoe Fluent Icons', 'Segoe MDL2 Assets';
  display: inline-flex;
}
.ppc-empty-title {
  color: var(--text-primary, #000);
}
.ppc-empty-message {
  color: var(--text-secondary, #555);
  max-width: 400px;
}
.ppc-empty-actions {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}
</style>