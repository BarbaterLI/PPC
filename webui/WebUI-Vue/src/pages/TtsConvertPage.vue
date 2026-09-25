<script setup lang="ts">
// TTS 转换页 —— 对齐 React `pages/TtsConvertPage.tsx`。
// 功能：配置表单创建 TTS 转换任务 → useSse 订阅进度流 → 展示进度/结果/取消。
import { computed, onBeforeUnmount, reactive, ref } from 'vue'
import {
  WinButton,
  WinComboBox,
  WinContentDialog,
  WinExpander,
  WinGrid,
  WinInfoBar,
  WinListView,
  WinProgressBar,
  WinProgressRing,
  WinSlider,
  WinStackPanel,
  WinTextBox,
  WinTextBlock,
  WinToggleSwitch,
} from 'winui-web-library'
import { PageHeader, EmptyState, LoadingSkeleton } from '@/components'
import { useApi } from '@/composables/useApi'
import { useSse } from '@/composables/useSse'
import { useToast } from '@/composables/useToast'
import { browseApi, systemApi, taskApi } from '@/api/client'
import type { VoiceGroup } from '@/api/types'

// ---- 类型 ----
interface SseProgressData {
  progress: number
  current_file?: string
  speed?: string
  eta?: string
  total_files?: number
  completed_files?: number
}

interface ConvertResult {
  success_count: number
  failure_count: number
  duration: number
  failed_items: { file: string; error: string }[]
}

type PageState = 'config' | 'running' | 'completed'
type PickerField = 'input_dir' | 'output_dir'

interface ConvertFormData {
  input_dir: string
  output_dir: string
  voice_id: string
  concurrency: number
  rate: string
  recursive: boolean
  resume: boolean
  tts_backend: string
  tts_proxy: string
}

const INITIAL_FORM: ConvertFormData = {
  input_dir: '',
  output_dir: '',
  voice_id: '',
  concurrency: 12,
  rate: '+0%',
  recursive: false,
  resume: false,
  tts_backend: 'auto',
  tts_proxy: '',
}

const CONNECTION_TIMEOUT = 300000

// ---- 状态 ----
const pageState = ref<PageState>('config')
const submitting = ref(false)
const submitError = ref<string | null>(null)
const taskId = ref<string | null>(null)
const progressData = ref<SseProgressData>({ progress: 0 })
const convertResult = ref<ConvertResult | null>(null)

const form = reactive<ConvertFormData>({ ...INITIAL_FORM })

let startTime = 0
let sseTimeout: ReturnType<typeof setTimeout> | null = null

const { showToast } = useToast()
const { connect: connectSse, close: closeSse } = useSse()

// 音色列表（useApi 拉取）
const { data: voiceGroups, loading: voicesLoading } = useApi<VoiceGroup[]>(() =>
  systemApi.getVoices(),
)

const voiceOptions = computed(() => {
  if (!voiceGroups.value || !Array.isArray(voiceGroups.value)) return []
  const options: { value: string; label: string }[] = []
  for (const group of voiceGroups.value) {
    if (!group.voices || !Array.isArray(group.voices)) continue
    for (const voice of group.voices) {
      options.push({ value: voice.name, label: `${voice.display_name} (${voice.locale})` })
    }
  }
  return options
})

const isFormValid = computed(
  () => Boolean(form.input_dir) && Boolean(form.output_dir) && Boolean(form.voice_id),
)

// ---- SSE 超时控制 ----
const resetSseTimeout = () => {
  if (sseTimeout) clearTimeout(sseTimeout)
  sseTimeout = setTimeout(() => {
    const currentTaskId = taskId.value
    closeSse()
    sseTimeout = null
    pageState.value = 'config'
    taskId.value = null
    showToast({
      title: '连接超时',
      body: '连接超时：任务可能已完成但未收到进度，已尝试取消；也可在任务列表查看',
      intent: 'warning',
    })
    if (currentTaskId) {
      taskApi.cancel(currentTaskId).catch((err) => console.warn('取消超时任务失败', err))
    }
  }, CONNECTION_TIMEOUT)
}

const clearSseTimeout = () => {
  if (sseTimeout) {
    clearTimeout(sseTimeout)
    sseTimeout = null
  }
}

onBeforeUnmount(() => {
  clearSseTimeout()
  closeSse()
})

// ---- 开始转换 ----
const handleStartConvert = async () => {
  if (!isFormValid.value) {
    submitError.value = '请填写输入目录、输出目录并选择语音'
    return
  }

  submitting.value = true
  submitError.value = null

  try {
    const result = await taskApi.createConvert({
      input_dir: form.input_dir,
      output_dir: form.output_dir,
      voice: form.voice_id,
      concurrency: form.concurrency,
      rate: form.rate,
      recursive: form.recursive,
      resume: form.resume,
      tts_backend: form.tts_backend,
      tts_proxy: form.tts_proxy,
    })
    if (result.success === false) {
      throw new Error(result.error?.message || '创建任务失败')
    }
    const id = result.data?.task_id
    if (!id) throw new Error('未获取到任务ID')

    taskId.value = id
    pageState.value = 'running'
    progressData.value = { progress: 0 }
    startTime = Date.now()

    resetSseTimeout()

    connectSse(taskApi.streamProgress(id), {
      progress: (event) => {
        resetSseTimeout()
        try {
          progressData.value = JSON.parse(event.data) as SseProgressData
        } catch (error) {
          console.warn('解析 progress 事件失败', error)
        }
      },
      complete: (event) => {
        clearSseTimeout()
        try {
          convertResult.value = JSON.parse(event.data) as ConvertResult
        } catch {
          convertResult.value = {
            success_count: 0,
            failure_count: 0,
            duration: (Date.now() - startTime) / 1000,
            failed_items: [],
          }
        }
        pageState.value = 'completed'
        closeSse()
      },
      error: () => {
        clearSseTimeout()
        convertResult.value =
          convertResult.value ?? {
            success_count: 0,
            failure_count: 0,
            duration: (Date.now() - startTime) / 1000,
            failed_items: [],
          }
        pageState.value = 'completed'
        closeSse()
      },
    })
  } catch (err: unknown) {
    submitError.value = err instanceof Error ? err.message : '启动转换任务失败'
  } finally {
    submitting.value = false
  }
}

// ---- 取消 / 重置 ----
const handleCancel = async () => {
  const id = taskId.value
  if (!id) return
  closeSse()
  try {
    await taskApi.cancel(id)
  } catch (error) {
    console.warn('取消任务失败', error)
  }
  pageState.value = 'config'
  taskId.value = null
  progressData.value = { progress: 0 }
}

const handleReset = () => {
  pageState.value = 'config'
  taskId.value = null
  progressData.value = { progress: 0 }
  convertResult.value = null
}

// ---- 目录选择（复用 browseApi，弹出内容对话框浏览服务器目录） ----
const pickerFor = ref<PickerField | null>(null)
const browseData = ref<BrowseDir | null>(null)
const browseLoading = ref(false)
const browseError = ref<string | null>(null)

interface BrowseDir {
  path: string
  parent: string
  home: string
  folders: string[]
  is_root: boolean
}

const openPicker = (field: PickerField) => {
  pickerFor.value = field
  void loadBrowse(undefined)
}
const closePicker = () => {
  pickerFor.value = null
  browseData.value = null
  browseError.value = null
}

const loadBrowse = async (path: string | undefined) => {
  browseLoading.value = true
  browseError.value = null
  try {
    const res = await browseApi.list(path)
    if (res.success === false) throw new Error(res.error?.message || '获取目录失败')
    browseData.value = res.data ?? null
  } catch (err: unknown) {
    browseError.value = err instanceof Error ? err.message : '获取目录失败'
  } finally {
    browseLoading.value = false
  }
}

const navigateFolder = (folder: string) => {
  if (!browseData.value) return
  void loadBrowse(`${browseData.value.path}/${folder}`.replace(/\/+/g, '/'))
}

const navigateUp = () => {
  if (!browseData.value?.parent) return
  void loadBrowse(browseData.value.parent)
}

const chooseCurrent = () => {
  if (!pickerFor.value || !browseData.value) return
  form[pickerFor.value] = browseData.value.path
  closePicker()
}

// ---- 工具 ----
const formatDuration = (seconds: number): string => {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}
</script>

<template>
  <div class="ppc-page">
    <div class="ppc-page-container">
      <PageHeader title="TTS 转换" description="文本转语音批量转换工具" />

      <div class="ppc-tts-content">
        <!-- 配置表单 -->
        <div v-if="pageState === 'config'" class="ppc-card">
          <WinTextBlock FontSize="18" FontWeight="600" :Text="'转换配置'" />

          <!-- 目录行 -->
          <div class="ppc-form-grid">
            <div class="ppc-form-group">
              <WinTextBlock class="ppc-form-label" :Text="'输入目录'" />
              <WinTextBlock class="ppc-form-desc" FontSize="12" :Text="'包含待转换文本文件的目录路径'" />
              <div class="ppc-form-row">
                <WinTextBox
                  class="ppc-form-field"
                  :Text="form.input_dir"
                  :IsEnabled="!submitting"
                  PlaceholderText="/path/to/input"
                  :ShowDeleteButton="false"
                  @update:Text="(v: string) => (form.input_dir = v)"
                />
                <WinButton
                  Style="DefaultButtonStyle"
                  :IsEnabled="!submitting"
                  :Content="'浏览…'"
                  @Click="openPicker('input_dir')"
                />
              </div>
            </div>

            <div class="ppc-form-group">
              <WinTextBlock class="ppc-form-label" :Text="'输出目录'" />
              <WinTextBlock class="ppc-form-desc" FontSize="12" :Text="'生成的音频文件保存路径'" />
              <div class="ppc-form-row">
                <WinTextBox
                  class="ppc-form-field"
                  :Text="form.output_dir"
                  :IsEnabled="!submitting"
                  PlaceholderText="/path/to/output"
                  :ShowDeleteButton="false"
                  @update:Text="(v: string) => (form.output_dir = v)"
                />
                <WinButton
                  Style="DefaultButtonStyle"
                  :IsEnabled="!submitting"
                  :Content="'浏览…'"
                  @Click="openPicker('output_dir')"
                />
              </div>
            </div>

            <!-- 语音 -->
            <div class="ppc-form-group">
              <WinTextBlock class="ppc-form-label" :Text="'语音选择'" />
              <WinTextBlock class="ppc-form-desc" FontSize="12" :Text="'选择要使用的语音角色'" />
              <WinComboBox
                class="ppc-form-field"
                :ItemsSource="voiceOptions"
                DisplayMemberPath="label"
                SelectedValuePath="value"
                :SelectedValue="form.voice_id"
                :PlaceholderText="voicesLoading ? '加载中...' : '选择语音'"
                :IsEnabled="!submitting && !voicesLoading"
                :Header="voicesLoading ? '' : ''"
                @update:SelectedValue="(v: unknown) => (form.voice_id = typeof v === 'string' ? v : '')"
              />
            </div>

            <!-- 并发数 -->
            <div class="ppc-form-group">
              <WinTextBlock class="ppc-form-label" :Text="`并发数: ${form.concurrency}`" />
              <WinTextBlock class="ppc-form-desc" FontSize="12" :Text="'同时处理的文件数量 (1-64)'" />
              <WinSlider
                :Minimum="1"
                :Maximum="64"
                :StepFrequency="1"
                Width="100%"
                :IsEnabled="!submitting"
                :modelValue="form.concurrency"
                @update:modelValue="(v: number) => (form.concurrency = v)"
              />
            </div>

            <!-- 语速 -->
            <div class="ppc-form-group">
              <WinTextBlock class="ppc-form-label" :Text="'语速'" />
              <WinTextBlock class="ppc-form-desc" FontSize="12" :Text="'语音播放速度调节'" />
              <WinTextBox
                :Text="form.rate"
                :IsEnabled="!submitting"
                PlaceholderText="+0%"
                :ShowDeleteButton="false"
                @update:Text="(v: string) => (form.rate = v)"
              />
            </div>

            <!-- 合成后端 -->
            <div class="ppc-form-group">
              <WinTextBlock class="ppc-form-label" :Text="'合成后端'" />
              <WinTextBlock class="ppc-form-desc" FontSize="12" :Text="'TTS 合成后端来源'" />
              <WinComboBox
                :ItemsSource="['auto', 'builtin', 'env']"
                :SelectedValue="form.tts_backend"
                :IsEnabled="!submitting"
                @update:SelectedValue="(v: unknown) => (form.tts_backend = typeof v === 'string' ? v : 'auto')"
              />
            </div>

            <!-- 代理地址 -->
            <div class="ppc-form-group ppc-form-span2">
              <WinTextBlock class="ppc-form-label" :Text="'代理地址'" />
              <WinTextBlock class="ppc-form-desc" FontSize="12" :Text="'可选：HTTP 代理，如 http://127.0.0.1:7890'" />
              <WinTextBox
                :Text="form.tts_proxy"
                :IsEnabled="!submitting"
                PlaceholderText="http://127.0.0.1:7890"
                :ShowDeleteButton="false"
                @update:Text="(v: string) => (form.tts_proxy = v)"
              />
            </div>
          </div>

          <!-- 开关行 -->
          <div class="ppc-form-switches">
            <WinToggleSwitch
              :IsOn="form.recursive"
              :IsEnabled="!submitting"
              :OnContent="'递归处理子目录'"
              :OffContent="'递归处理子目录'"
              @update:IsOn="(v: boolean) => (form.recursive = v)"
            />
            <WinToggleSwitch
              :IsOn="form.resume"
              :IsEnabled="!submitting"
              :OnContent="'启用断点续传'"
              :OffContent="'启用断点续传'"
              @update:IsOn="(v: boolean) => (form.resume = v)"
            />
          </div>

          <WinInfoBar
            v-if="submitError"
            Severity="Error"
            :IsOpen="Boolean(submitError)"
            Title="提交失败"
            :Message="submitError"
          />

          <div class="ppc-form-actions">
            <WinButton
              Style="AccentButtonStyle"
              :IsEnabled="isFormValid && !submitting"
              @Click="handleStartConvert"
            >
              <span class="ppc-btn-with-ring">
                <WinProgressRing
                  v-if="submitting"
                  :IsActive="true"
                  :IsIndeterminate="true"
                  :Width="16"
                  :Height="16"
                />
                <span>{{ submitting ? '正在提交...' : '开始转换' }}</span>
              </span>
            </WinButton>
          </div>
        </div>

        <!-- 进度区 -->
        <div v-else-if="pageState === 'running'" class="ppc-card">
          <div class="ppc-progress-header">
            <div class="ppc-progress-title">
              <WinProgressRing :IsActive="true" :IsIndeterminate="true" :Width="20" :Height="20" />
              <WinTextBlock FontSize="18" FontWeight="600" :Text="'转换进行中'" />
            </div>
            <WinTextBlock class="ppc-progress-percent" FontSize="24" FontWeight="600">
              {{ Math.round(progressData.progress) }}%
            </WinTextBlock>
          </div>

          <WinProgressBar
            :Value="progressData.progress"
            :Maximum="100"
            :Width="'100%'"
            :Height="4"
          />

          <WinGrid ColumnSpacing="16" RowSpacing="12" ColumnDefinitions="1fr,1fr,1fr">
            <div class="ppc-progress-detail">
              <WinTextBlock class="ppc-progress-label" FontSize="12" :Text="'当前文件'" />
              <WinTextBlock class="ppc-progress-value" FontSize="14" FontWeight="600">
                {{ progressData.current_file || '-' }}
              </WinTextBlock>
            </div>
            <div class="ppc-progress-detail">
              <WinTextBlock class="ppc-progress-label" FontSize="12" :Text="'速度'" />
              <WinTextBlock class="ppc-progress-value" FontSize="14" FontWeight="600">
                {{ progressData.speed || '-' }}
              </WinTextBlock>
            </div>
            <div class="ppc-progress-detail">
              <WinTextBlock class="ppc-progress-label" FontSize="12" :Text="'预计剩余时间'" />
              <WinTextBlock class="ppc-progress-value" FontSize="14" FontWeight="600">
                {{ progressData.eta || '-' }}
              </WinTextBlock>
            </div>
          </WinGrid>

          <div class="ppc-form-actions">
            <WinButton Style="DefaultButtonStyle" :Content="'取消转换'" @Click="handleCancel" />
          </div>
        </div>

        <!-- 结果区 -->
        <div v-else-if="pageState === 'completed' && convertResult" class="ppc-card">
          <WinTextBlock FontSize="18" FontWeight="600" :Text="'转换完成'" />

          <WinGrid ColumnSpacing="16" RowSpacing="12" ColumnDefinitions="1fr,1fr,1fr">
            <div class="ppc-result-stat">
              <WinTextBlock class="ppc-result-value ppc-stat-success" FontSize="28" FontWeight="600">
                {{ convertResult.success_count }}
              </WinTextBlock>
              <WinTextBlock class="ppc-result-label" FontSize="12" :Text="'成功'" />
            </div>
            <div class="ppc-result-stat">
              <WinTextBlock class="ppc-result-value ppc-stat-failure" FontSize="28" FontWeight="600">
                {{ convertResult.failure_count }}
              </WinTextBlock>
              <WinTextBlock class="ppc-result-label" FontSize="12" :Text="'失败'" />
            </div>
            <div class="ppc-result-stat">
              <WinTextBlock class="ppc-result-value ppc-stat-accent" FontSize="28" FontWeight="600">
                {{ formatDuration(convertResult.duration) }}
              </WinTextBlock>
              <WinTextBlock class="ppc-result-label" FontSize="12" :Text="'耗时'" />
            </div>
          </WinGrid>

          <WinExpander
            v-if="convertResult.failed_items.length > 0"
            :Header="`失败项目 (${convertResult.failed_items.length})`"
          >
            <WinStackPanel Spacing="8">
              <div v-for="(item, idx) in convertResult.failed_items" :key="idx" class="ppc-failed-item">
                <WinTextBlock class="ppc-failed-name" FontSize="14" FontWeight="600" :Text="item.file" />
                <WinTextBlock class="ppc-failed-err" FontSize="12" :Text="item.error" />
              </div>
            </WinStackPanel>
          </WinExpander>

          <div class="ppc-form-actions">
            <WinButton Style="AccentButtonStyle" :Content="'开始新转换'" @Click="handleReset" />
          </div>
        </div>
      </div>
    </div>

    <!-- 目录选择对话框 -->
    <WinContentDialog
      :IsOpen="pickerFor !== null"
      :Title="pickerFor === 'input_dir' ? '选择输入目录' : '选择输出目录'"
      :IsLightDismissEnabled="false"
      @update:IsOpen="(v: boolean) => !v && closePicker()"
    >
      <WinStackPanel Spacing="12">
        <div class="ppc-browse-current">
          <WinTextBlock class="ppc-browse-path" FontFamily="Consolas, monospace" :Text="browseData?.path ?? '…'" />
          <WinButton
            Style="SubtleButtonStyle"
            :IsEnabled="Boolean(browseData?.parent && !browseData.is_root)"
            :Content="'上一级'"
            @Click="navigateUp"
          />
        </div>

        <WinInfoBar
          v-if="browseError"
          Severity="Error"
          :IsOpen="Boolean(browseError)"
          Title="加载失败"
          :Message="browseError"
        />

        <LoadingSkeleton v-if="browseLoading" :count="5" variant="line" />
        <WinListView
          v-else-if="browseData && browseData.folders.length > 0"
          class="ppc-browse-list"
          :ItemsSource="browseData.folders"
          SelectionMode="None"
        >
          <template #item="{ item }">
            <WinButton Style="SubtleButtonStyle" HorizontalAlignment="Left" @Click="navigateFolder(item)">
              <span class="ppc-folder-glyph" aria-hidden="true">&#xE8B7;</span>
              <span>{{ item }}</span>
            </WinButton>
          </template>
        </WinListView>
        <EmptyState v-else :message="'当前目录没有子文件夹'" />

        <div class="ppc-form-actions">
          <WinButton
            Style="AccentButtonStyle"
            :IsEnabled="Boolean(browseData)"
            :Content="'使用此目录'"
            @Click="chooseCurrent"
          />
          <WinButton Style="SubtleButtonStyle" :Content="'取消'" @Click="closePicker" />
        </div>
      </WinStackPanel>
    </WinContentDialog>
  </div>
</template>

<style scoped>
.ppc-tts-content {
  display: flex;
  flex-direction: column;
  gap: 24px;
  width: 100%;
}
.ppc-form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.ppc-form-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.ppc-form-span2 {
  grid-column: 1 / -1;
}
.ppc-form-label {
  color: var(--text-primary);
}
.ppc-form-desc {
  color: var(--text-secondary);
}
.ppc-form-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.ppc-form-field {
  flex: 1;
  min-width: 0;
}
.ppc-form-switches {
  display: flex;
  align-items: center;
  gap: 24px;
  flex-wrap: wrap;
}
.ppc-form-actions {
  display: flex;
  align-items: flex-end;
  gap: 12px;
  padding-top: 8px;
}
.ppc-btn-with-ring {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.ppc-progress-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.ppc-progress-title {
  display: flex;
  align-items: center;
  gap: 8px;
}
.ppc-progress-percent {
  color: var(--text-primary);
}
.ppc-progress-detail {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.ppc-progress-label {
  color: var(--text-secondary);
}
.ppc-progress-value {
  color: var(--text-primary);
  word-break: break-all;
}
.ppc-result-stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 24px 16px;
  background-color: var(--card-bg);
  border: 1px solid var(--ctrl-border);
  border-radius: 6px;
}
.ppc-result-value {
  font-weight: 600;
}
.ppc-result-label {
  color: var(--text-secondary);
}
.ppc-stat-success {
  color: var(--ppc-status-success, #0f7b0f);
}
.ppc-stat-failure {
  color: var(--ppc-status-danger, #c42b1c);
}
.ppc-stat-accent {
  color: var(--accent-text);
}
.ppc-failed-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 16px;
  background-color: var(--card-bg);
  border: 1px solid var(--ctrl-border);
  border-radius: 6px;
}
.ppc-failed-name {
  color: var(--ppc-status-danger, #c42b1c);
}
.ppc-failed-err {
  color: var(--text-secondary);
  word-break: break-all;
}
.ppc-browse-current {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.ppc-browse-path {
  color: var(--text-primary);
  word-break: break-all;
}
.ppc-browse-list {
  max-height: 260px;
}
.ppc-folder-glyph {
  font-family: 'WinUIOnWebIcons', 'Segoe Fluent Icons', 'Segoe MDL2 Assets';
  margin-right: 8px;
  color: var(--accent-text);
}

@media (max-width: 767px) {
  .ppc-form-grid {
    grid-template-columns: 1fr;
  }
  .ppc-progress-detail {
    grid-column: auto;
  }
}
</style>