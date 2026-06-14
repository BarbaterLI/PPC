import { useState } from 'react'
import {
  tokens,
  Text,
  Button,
  Dropdown,
  Option,
  Dialog,
  DialogSurface,
  DialogBody,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@fluentui/react-components'

interface ConfigResetDialogProps {
  open: boolean
  onClose: () => void
  onReset: (preset: string) => Promise<void>
  resetting: boolean
}

export function ConfigResetDialog({
  open,
  onClose,
  onReset,
  resetting,
}: ConfigResetDialogProps) {
  const [preset, setPreset] = useState('balanced')

  const handleReset = async () => {
    await onReset(preset)
  }

  return (
    <Dialog open={open} onOpenChange={(_, data) => !data.open && onClose()}>
      <DialogSurface>
        <DialogBody>
          <DialogTitle>重置配置</DialogTitle>
          <DialogContent>
            <div style={{ display: 'flex', flexDirection: 'column', gap: tokens.spacingVerticalM }}>
              <Text>选择预设配置方案以重置所有设置：</Text>
              <Dropdown
                value={preset}
                selectedOptions={[preset]}
                onOptionSelect={(_, data) => setPreset(data.optionValue ?? 'balanced')}
                style={{ minWidth: '200px' }}
              >
                <Option value="balanced">均衡模式</Option>
                <Option value="speed">速度优先</Option>
                <Option value="quality">质量优先</Option>
              </Dropdown>
            </div>
          </DialogContent>
          <DialogActions>
            <Button appearance="secondary" onClick={onClose}>
              取消
            </Button>
            <Button appearance="primary" onClick={handleReset} disabled={resetting}>
              确认重置
            </Button>
          </DialogActions>
        </DialogBody>
      </DialogSurface>
    </Dialog>
  )
}
