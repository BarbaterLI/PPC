<script setup lang="ts">
// 配置项编辑单元：按类型渲染不同控件，并在被修改时高亮整行。
// 对齐 React `src/components/config/ConfigField.tsx`。
import { computed } from 'vue'
import {
  WinTextBlock,
  WinToggleSwitch,
  WinComboBox,
  WinNumberBox,
  WinSlider,
  WinTextBox,
} from 'winui-web-library'
import type { PPCConfig } from '@/api/types'

const props = defineProps<{
  config: PPCConfig
  value: string
  isChanged: boolean
  onChange: (value: string) => void
}>()

// 不可编辑/只读配置项禁用输入控件
const enabled = computed(() => props.config.editable !== false)
const isBooleanOn = computed(() => props.value === 'true' || props.value === '1')
const numValue = computed(() => Number(props.value) || 0)

// number + select 类型若带有 options 选项，则渲染为下拉框
const hasOptions = computed(() => Array.isArray(props.config.options) && props.config.options.length > 0)
</script>

<template>
  <div class="ppc-cfg-row" :class="{ 'ppc-cfg-row--changed': isChanged }">
    <div class="ppc-cfg-info">
      <div class="ppc-cfg-label">
        {{ config.key }}
        <span v-if="config.required" class="ppc-cfg-required">*</span>
      </div>
      <WinTextBlock v-if="config.description" class="ppc-cfg-desc" :Text="config.description" />
    </div>

    <div class="ppc-cfg-control">
      <!-- 布尔 -->
      <WinToggleSwitch
        v-if="config.type === 'boolean'"
        :IsOn="isBooleanOn"
        :IsEnabled="enabled"
        OnContent="启用"
        OffContent="禁用"
        @update:IsOn="(v: boolean) => onChange(v ? 'true' : 'false')"
      />

      <!-- 数字 + options：下拉 -->
      <WinComboBox
        v-else-if="config.type === 'number' && hasOptions"
        :ItemsSource="config.options"
        :SelectedValue="value"
        :IsEnabled="enabled"
        :MinWidth="200"
        @update:SelectedValue="(v: unknown) => onChange(v == null ? value : String(v))"
      />

      <!-- 数字：输入 + 滑杆 -->
      <div v-else-if="config.type === 'number'" class="ppc-cfg-number-row">
        <WinNumberBox
          :Value="numValue"
          :IsEnabled="enabled"
          :Width="120"
          SpinButtonPlacementMode="Inline"
          @update:Value="(v: number) => onChange(String(v))"
        />
        <WinSlider
          class="ppc-cfg-slider"
          :Value="numValue"
          :IsEnabled="enabled"
          :Minimum="0"
          :Maximum="1000"
          @update:Value="(v: number) => onChange(String(v))"
        />
      </div>

      <!-- 选择 -->
      <WinComboBox
        v-else-if="config.type === 'select'"
        :ItemsSource="config.options || []"
        :SelectedValue="value"
        :IsEnabled="enabled"
        :MinWidth="200"
        @update:SelectedValue="(v: unknown) => onChange(v == null ? value : String(v))"
      />

      <!-- 字符串 / 默认：文本框 -->
      <WinTextBox
        v-else
        :Text="value"
        :IsEnabled="enabled"
        :MinWidth="200"
        @update:Text="(v: string) => onChange(v)"
      />
    </div>
  </div>
</template>

<style scoped>
.ppc-cfg-row {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  padding: 10px 12px;
  border-radius: 6px;
}
.ppc-cfg-row--changed {
  background: var(--card-bg, #f3f3f3);
}
.ppc-cfg-info {
  flex: 1;
  min-width: 0;
}
.ppc-cfg-label {
  color: var(--text-primary, #000);
  font-weight: 600;
  font-size: 14px;
}
.ppc-cfg-required {
  color: #c42b1c;
  margin-left: 2px;
}
.ppc-cfg-desc {
  color: var(--text-secondary, #555);
  font-size: 12px;
  margin-top: 4px;
}
.ppc-cfg-control {
  flex: 2;
  min-width: 0;
}
.ppc-cfg-number-row {
  display: flex;
  align-items: center;
  gap: 16px;
}
.ppc-cfg-slider {
  flex: 1;
  min-width: 100px;
}
</style>