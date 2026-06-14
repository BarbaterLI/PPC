import {
  makeStyles,
  tokens,
  shorthands,
  Input,
  Switch,
  Slider,
  Dropdown,
  Option,
  Field,
  mergeClasses,
} from '@fluentui/react-components'
import type { PPC10Config } from '@/types'

const useStyles = makeStyles({
  fieldRow: {
    display: 'flex',
    alignItems: 'flex-start',
    ...shorthands.gap(tokens.spacingHorizontalM),
  },
  fieldInfo: {
    flex: 1,
    minWidth: 0,
  },
  fieldControl: {
    flex: 2,
    minWidth: 0,
  },
  fieldLabel: {
    fontWeight: tokens.fontWeightSemibold,
    fontSize: tokens.fontSizeBase300,
    color: tokens.colorNeutralForeground1,
  },
  fieldDesc: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
    marginTop: tokens.spacingVerticalXS,
  },
  fieldRequired: {
    color: tokens.colorPaletteRedForeground1,
    marginLeft: tokens.spacingHorizontalXS,
  },
  changed: {
    backgroundColor: tokens.colorPaletteYellowBackground1,
    ...shorthands.borderRadius(tokens.borderRadiusMedium),
    ...shorthands.padding(tokens.spacingVerticalS, tokens.spacingHorizontalS),
    marginLeft: `-${tokens.spacingHorizontalS}`,
    marginRight: `-${tokens.spacingHorizontalS}`,
  },
})

interface ConfigFieldProps {
  config: PPC10Config
  value: string
  isChanged: boolean
  onChange: (value: string) => void
}

export function ConfigField({
  config,
  value,
  isChanged,
  onChange,
}: ConfigFieldProps) {
  const styles = useStyles()

  const renderControl = () => {
    switch (config.type) {
      case 'boolean':
        return (
          <Switch
            checked={value === 'true' || value === '1'}
            onChange={(_, data) =>
              onChange(data.checked ? 'true' : 'false')
            }
            label={value === 'true' || value === '1' ? '启用' : '禁用'}
          />
        )
      case 'number': {
        const numVal = Number(value) || 0
        const hasOptions = config.options && config.options.length > 0
        if (hasOptions) {
          return (
            <Dropdown
              value={value}
              selectedOptions={[value]}
              onOptionSelect={(_, data) =>
                onChange(data.optionValue ?? value)
              }
              style={{ minWidth: '200px' }}
            >
              {config.options!.map((opt) => (
                <Option key={opt} value={opt}>
                  {opt}
                </Option>
              ))}
            </Dropdown>
          )
        }
        return (
          <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacingHorizontalM }}>
            <Input
              type="number"
              value={value}
              onChange={(_, data) => onChange(data.value)}
              style={{ width: '120px' }}
              min={0}
              max={10000}
            />
            <Slider
              value={numVal}
              min={0}
              max={1000}
              onChange={(_, data) => onChange(String(data.value))}
              style={{ flex: 1, minWidth: '100px' }}
            />
          </div>
        )
      }
      case 'select':
        return (
          <Dropdown
            value={value}
            selectedOptions={[value]}
            onOptionSelect={(_, data) =>
              onChange(data.optionValue ?? value)
            }
            style={{ minWidth: '200px' }}
          >
            {(config.options || []).map((opt) => (
              <Option key={opt} value={opt}>
                {opt}
              </Option>
            ))}
          </Dropdown>
        )
      default:
        return (
          <Input
            value={value}
            onChange={(_, data) => onChange(data.value)}
            style={{ minWidth: '200px' }}
          />
        )
    }
  }

  return (
    <div
      className={mergeClasses(styles.fieldRow, isChanged && styles.changed)}
    >
      <div className={styles.fieldInfo}>
        <div className={styles.fieldLabel}>
          {config.key}
          {config.required && <span className={styles.fieldRequired}>*</span>}
        </div>
        {config.description && (
          <div className={styles.fieldDesc}>{config.description}</div>
        )}
      </div>
      <div className={styles.fieldControl}>
        <Field>
          {renderControl()}
        </Field>
      </div>
    </div>
  )
}
