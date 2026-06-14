import {
  makeStyles,
  tokens,
  shorthands,
  Text,
  Button,
  Input,
  Dropdown,
  Option,
  Slider,
  Switch,
  Card,
  Spinner,
} from '@fluentui/react-components'
import {
  Play24Regular,
  FolderOpen24Regular,
} from '@fluentui/react-icons'

const MOBILE = '@media (max-width: 767px)'

const useStyles = makeStyles({
  formCard: {
    ...shorthands.padding(tokens.spacingVerticalL, tokens.spacingHorizontalL),
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalL),
  },
  formGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    ...shorthands.gap(tokens.spacingHorizontalM, tokens.spacingVerticalM),
    [MOBILE]: {
      gridTemplateColumns: '1fr',
    },
  },
  formGridFull: {
    gridColumn: '1 / -1',
  },
  fieldGroup: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalXS),
  },
  fieldLabel: {
    fontSize: tokens.fontSizeBase300,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
  },
  fieldDescription: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
  },
  fieldInput: {
    width: '100%',
  },
  formActions: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalM),
    paddingTop: tokens.spacingVerticalS,
  },
  errorText: {
    color: tokens.colorPaletteRedForeground1,
    fontSize: tokens.fontSizeBase300,
  },
})

interface ConvertFormData {
  input_dir: string
  output_dir: string
  voice_id: string
  concurrency: number
  rate: string
  recursive: boolean
  resume: boolean
}

interface TtsConvertConfigPanelProps {
  formData: ConvertFormData
  setFormData: (data: ConvertFormData) => void
  voiceOptions: { value: string; label: string }[]
  voicesLoading: boolean
  submitting: boolean
  submitError: string | null
  onSubmit: () => void
}

export function TtsConvertConfigPanel({
  formData,
  setFormData,
  voiceOptions,
  voicesLoading,
  submitting,
  submitError,
  onSubmit,
}: TtsConvertConfigPanelProps) {
  const styles = useStyles()

  const updateField = <K extends keyof ConvertFormData>(
    field: K,
    value: ConvertFormData[K],
  ) => {
    setFormData({ ...formData, [field]: value })
  }

  const isFormValid = formData.input_dir && formData.output_dir && formData.voice_id

  return (
    <Card className={styles.formCard}>
      <Text size={400} weight="semibold">转换配置</Text>

      <div className={styles.formGrid}>
        <div className={styles.fieldGroup}>
          <Text className={styles.fieldLabel}>输入目录</Text>
          <Text size={200} className={styles.fieldDescription}>包含待转换文本文件的目录路径</Text>
          <Input
            className={styles.fieldInput}
            value={formData.input_dir}
            onChange={(_, data) => updateField('input_dir', data.value)}
            placeholder="/path/to/input"
            contentBefore={<FolderOpen24Regular />}
            disabled={submitting}
          />
        </div>

        <div className={styles.fieldGroup}>
          <Text className={styles.fieldLabel}>输出目录</Text>
          <Text size={200} className={styles.fieldDescription}>生成的音频文件保存路径</Text>
          <Input
            className={styles.fieldInput}
            value={formData.output_dir}
            onChange={(_, data) => updateField('output_dir', data.value)}
            placeholder="/path/to/output"
            contentBefore={<FolderOpen24Regular />}
            disabled={submitting}
          />
        </div>

        <div className={styles.fieldGroup}>
          <Text className={styles.fieldLabel}>语音选择</Text>
          <Text size={200} className={styles.fieldDescription}>选择要使用的语音角色</Text>
          <Dropdown
            className={styles.fieldInput}
            value={formData.voice_id}
            selectedOptions={formData.voice_id ? [formData.voice_id] : []}
            onOptionSelect={(_, data) =>
              updateField('voice_id', data.optionValue ?? '')
            }
            placeholder={voicesLoading ? '加载中...' : '选择语音'}
            disabled={submitting || voicesLoading}
          >
            {voiceOptions.map((opt) => (
              <Option key={opt.value} value={opt.value}>
                {opt.label}
              </Option>
            ))}
          </Dropdown>
        </div>

        <div className={styles.fieldGroup}>
          <Text className={styles.fieldLabel}>
            并发数: {formData.concurrency}
          </Text>
          <Text size={200} className={styles.fieldDescription}>同时处理的文件数量 (1-64)</Text>
          <Slider
            className={styles.fieldInput}
            value={formData.concurrency}
            min={1}
            max={64}
            step={1}
            onChange={(_, data) => updateField('concurrency', data.value)}
            disabled={submitting}
          />
        </div>

        <div className={styles.fieldGroup}>
          <Text className={styles.fieldLabel}>语速</Text>
          <Text size={200} className={styles.fieldDescription}>语音播放速度调节</Text>
          <Input
            className={styles.fieldInput}
            value={formData.rate}
            onChange={(_, data) => updateField('rate', data.value)}
            placeholder="+0%"
            disabled={submitting}
          />
        </div>

        <div className={styles.formGridFull}>
          <div className={styles.formActions}>
            <Switch
              label="递归处理子目录"
              checked={formData.recursive}
              onChange={(_, data) => updateField('recursive', data.checked)}
              disabled={submitting}
            />
            <Switch
              label="启用断点续传"
              checked={formData.resume}
              onChange={(_, data) => updateField('resume', data.checked)}
              disabled={submitting}
            />
          </div>
        </div>
      </div>

      {submitError && (
        <Text className={styles.errorText}>{submitError}</Text>
      )}

      <div className={styles.formActions}>
        <Button
          appearance="primary"
          icon={submitting ? <Spinner size="tiny" /> : <Play24Regular />}
          onClick={onSubmit}
          disabled={!isFormValid || submitting}
        >
          {submitting ? '正在提交...' : '开始转换'}
        </Button>
      </div>
    </Card>
  )
}
