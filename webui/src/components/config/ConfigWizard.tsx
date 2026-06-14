import { useState, useCallback, useMemo } from 'react'
import {
  makeStyles,
  tokens,
  shorthands,
  Button,
  Input,
  Slider,
  Dropdown,
  Option,
  Dialog,
  DialogSurface,
  DialogBody,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@fluentui/react-components'

const VOICE_OPTIONS = ['zh-CN-XiaoxiaoNeural', 'zh-CN-YunxiNeural', 'zh-CN-YunyangNeural', 'zh-CN-XiaoyiNeural']
const FORMAT_OPTIONS = ['mp3', 'wav', 'ogg', 'aac']

const useStyles = makeStyles({
  wizardStep: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalL),
    minHeight: '200px',
  },
  wizardStepTitle: {
    fontWeight: tokens.fontWeightSemibold,
    fontSize: tokens.fontSizeBase400,
  },
})

interface WizardData {
  voice: string
  concurrency: number
  outputFormat: string
}

interface ConfigWizardProps {
  open: boolean
  onClose: () => void
  onComplete: (data: WizardData) => void
}

export function ConfigWizard({
  open,
  onClose,
  onComplete,
}: ConfigWizardProps) {
  const styles = useStyles()
  const [step, setStep] = useState(0)
  const [data, setData] = useState<WizardData>({
    voice: 'zh-CN-XiaoxiaoNeural',
    concurrency: 4,
    outputFormat: 'mp3',
  })

  const canProceed = useMemo(() => {
    if (step === 0) return !!data.voice
    if (step === 1) return data.concurrency > 0
    return true
  }, [step, data.voice, data.concurrency])

  const handleNext = useCallback(() => {
    if (step < 2) {
      setStep((s) => s + 1)
    } else {
      onComplete(data)
    }
  }, [step, data, onComplete])

  const handlePrev = useCallback(() => {
    if (step > 0) setStep((s) => s - 1)
  }, [step])

  return (
    <Dialog open={open} onOpenChange={(_, data) => !data.open && onClose()}>
      <DialogSurface>
        <DialogBody>
          <DialogTitle>配置向导</DialogTitle>
          <DialogContent>
            <div className={styles.wizardStep}>
              {step === 0 && (
                <>
                  <div className={styles.wizardStepTitle}>步骤 1/3：选择默认语音</div>
                  <Dropdown
                    value={data.voice}
                    selectedOptions={[data.voice]}
                    onOptionSelect={(_, d) =>
                      setData((prev) => ({ ...prev, voice: d.optionValue ?? prev.voice }))
                    }
                    style={{ minWidth: '250px' }}
                  >
                    {VOICE_OPTIONS.map((v) => (
                      <Option key={v} value={v}>
                        {v}
                      </Option>
                    ))}
                  </Dropdown>
                </>
              )}
              {step === 1 && (
                <>
                  <div className={styles.wizardStepTitle}>步骤 2/3：设置并发数</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacingHorizontalM }}>
                    <Input
                      type="number"
                      value={String(data.concurrency)}
                      onChange={(_, d) =>
                        setData((prev) => ({ ...prev, concurrency: Number(d.value) || 1 }))
                      }
                      style={{ width: '100px' }}
                    />
                    <Slider
                      value={data.concurrency}
                      min={1}
                      max={16}
                      step={1}
                      onChange={(_, d) =>
                        setData((prev) => ({ ...prev, concurrency: d.value }))
                      }
                      style={{ flex: 1, minWidth: '150px' }}
                    />
                  </div>
                </>
              )}
              {step === 2 && (
                <>
                  <div className={styles.wizardStepTitle}>步骤 3/3：选择输出格式</div>
                  <Dropdown
                    value={data.outputFormat}
                    selectedOptions={[data.outputFormat]}
                    onOptionSelect={(_, d) =>
                      setData((prev) => ({
                        ...prev,
                        outputFormat: d.optionValue ?? prev.outputFormat,
                      }))
                    }
                    style={{ minWidth: '200px' }}
                  >
                    {FORMAT_OPTIONS.map((f) => (
                      <Option key={f} value={f}>
                        {f.toUpperCase()}
                      </Option>
                    ))}
                  </Dropdown>
                </>
              )}
            </div>
          </DialogContent>
          <DialogActions>
            <Button
              appearance="secondary"
              onClick={handlePrev}
              disabled={step === 0}
            >
              上一步
            </Button>
            <Button appearance="primary" onClick={handleNext} disabled={!canProceed}>
              {step === 2 ? '完成' : '下一步'}
            </Button>
          </DialogActions>
        </DialogBody>
      </DialogSurface>
    </Dialog>
  )
}
