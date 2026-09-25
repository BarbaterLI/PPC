<script setup lang="ts">
// 配置向导：分 6 步引导用户设置 TTS 转换默认参数。
// 对齐 React `src/components/config/ConfigWizard.tsx`。
import { ref, watch } from 'vue'
import { WinContentDialog, WinTextBlock, WinComboBox, WinNumberBox, WinSlider, WinTextBox } from 'winui-web-library'
import { systemApi } from '@/api/client'

const DEFAULT_VOICES = ['zh-CN-XiaoxiaoNeural', 'zh-CN-YunxiNeural', 'zh-CN-YunyangNeural', 'zh-CN-XiaoyiNeural']
const FORMAT_OPTIONS = ['mp3', 'wav', 'ogg', 'aac']
const BACKEND_OPTIONS = ['auto', 'builtin', 'env']
const TOTAL_STEPS = 6

interface WizardData {
  voice: string
  concurrency: number
  rate: string
  maxRetries: number
  outputFormat: string
  outputBackend: string
}

const props = defineProps<{ open: boolean }>()
const emit = defineEmits<{
  close: []
  complete: [data: WizardData]
}>()

const step = ref(0)
const data = ref<WizardData>({
  voice: 'zh-CN-XiaoxiaoNeural',
  concurrency: 4,
  rate: '+0%',
  maxRetries: 3,
  outputFormat: 'mp3',
  outputBackend: 'auto',
})
const voiceOptions = ref<string[]>([...DEFAULT_VOICES])

// 打开时拉取可用语音
watch(
  () => props.open,
  (open) => {
    if (!open) return
    void systemApi
      .getVoices()
      .then((res) => {
        if (res.success === false) return
        const names: string[] = []
        for (const group of res.data ?? []) {
          for (const v of group.voices ?? []) {
            if (v?.name && !names.includes(v.name)) names.push(v.name)
          }
        }
        if (names.length > 0) voiceOptions.value = names
      })
      .catch(() => {})
  },
)

// 步骤前进条件
const _canProceed = () => {
  if (step.value === 0) return Boolean(data.value.voice)
  if (step.value === 1) return data.value.concurrency > 0
  if (step.value === 2) return data.value.rate.trim().length > 0
  return true
}
const isLastStep = () => step.value === TOTAL_STEPS - 1

const handlePrev = () => {
  if (step.value > 0) step.value -= 1
}
const handleNext = () => {
  if (step.value < TOTAL_STEPS - 1) {
    step.value += 1
  } else {
    emit('complete', { ...data.value })
  }
}

const onCloseButton = () => {
  emit('close')
}

// 关闭后复位步骤状态
watch(
  () => props.open,
  (open) => {
    if (!open) step.value = 0
  },
)
</script>

<template>
  <WinContentDialog
    :IsOpen="props.open"
    Title="配置向导"
    DefaultButton="Primary"
    :PrimaryButtonText="isLastStep() ? '完成' : '下一步'"
    :SecondaryButtonText="'上一步'"
    @PrimaryButtonClick="handleNext"
    @SecondaryButtonClick="handlePrev"
    @CloseButtonClick="onCloseButton"
  >
    <div class="ppc-wizard-step">
      <template v-if="step === 0">
        <WinTextBlock class="ppc-wizard-step-title" :Text="`步骤 1/${TOTAL_STEPS}：选择默认语音`" />
        <WinComboBox
          :ItemsSource="voiceOptions"
          :SelectedValue="data.voice"
          :MinWidth="260"
          @update:SelectedValue="(v: unknown) => (data.voice = v == null ? data.voice : String(v))"
        />
      </template>

      <template v-else-if="step === 1">
        <WinTextBlock class="ppc-wizard-step-title" :Text="`步骤 2/${TOTAL_STEPS}：设置并发数`" />
        <div class="ppc-wizard-row">
          <WinNumberBox
            :Value="data.concurrency"
            :Width="100"
            SpinButtonPlacementMode="Inline"
            @update:Value="(v: number) => (data.concurrency = v || 1)"
          />
          <WinSlider
            class="ppc-wizard-slider"
            :Value="data.concurrency"
            :Minimum="1"
            :Maximum="64"
            :StepFrequency="1"
            @update:Value="(v: number) => (data.concurrency = v)"
          />
        </div>
      </template>

      <template v-else-if="step === 2">
        <WinTextBlock class="ppc-wizard-step-title" :Text="`步骤 3/${TOTAL_STEPS}：语速`" />
        <WinTextBox
          :Text="data.rate"
          PlaceholderText="+0%"
          @update:Text="(v: string) => (data.rate = v)"
        />
      </template>

      <template v-else-if="step === 3">
        <WinTextBlock class="ppc-wizard-step-title" :Text="`步骤 4/${TOTAL_STEPS}：最大重试次数`" />
        <div class="ppc-wizard-row">
          <WinNumberBox
            :Value="data.maxRetries"
            :Width="100"
            SpinButtonPlacementMode="Inline"
            @update:Value="(v: number) => (data.maxRetries = v || 0)"
          />
          <WinSlider
            class="ppc-wizard-slider"
            :Value="data.maxRetries"
            :Minimum="0"
            :Maximum="10"
            :StepFrequency="1"
            @update:Value="(v: number) => (data.maxRetries = v)"
          />
        </div>
      </template>

      <template v-else-if="step === 4">
        <WinTextBlock class="ppc-wizard-step-title" :Text="`步骤 5/${TOTAL_STEPS}：输出格式`" />
        <WinComboBox
          :ItemsSource="FORMAT_OPTIONS"
          :SelectedValue="data.outputFormat"
          :MinWidth="200"
          @update:SelectedValue="(v: unknown) => (data.outputFormat = v == null ? data.outputFormat : String(v))"
        />
      </template>

      <template v-else-if="step === 5">
        <WinTextBlock class="ppc-wizard-step-title" :Text="`步骤 6/${TOTAL_STEPS}：TTS 合成后端`" />
        <WinComboBox
          :ItemsSource="BACKEND_OPTIONS"
          :SelectedValue="data.outputBackend"
          :MinWidth="220"
          @update:SelectedValue="(v: unknown) => (data.outputBackend = v == null ? data.outputBackend : String(v))"
        />
      </template>
    </div>
  </WinContentDialog>
</template>

<style scoped>
.ppc-wizard-step {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 200px;
}
.ppc-wizard-step-title {
  font-weight: 600;
  color: var(--text-primary, #000);
}
.ppc-wizard-row {
  display: flex;
  align-items: center;
  gap: 16px;
}
.ppc-wizard-slider {
  flex: 1;
  min-width: 150px;
}
</style>