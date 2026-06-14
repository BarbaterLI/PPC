import {
  makeStyles,
  tokens,
  shorthands,
  Button,
  Checkbox,
} from '@fluentui/react-components'
import {
  Play24Regular,
  History24Regular,
} from '@fluentui/react-icons'

const ANALYZER_MODULES = [
  { key: 'performance', label: '性能分析' },
  { key: 'config', label: '配置检查' },
  { key: 'errors', label: '错误诊断' },
  { key: 'dependency', label: '依赖分析' },
  { key: 'network', label: '网络检测' },
  { key: 'resource', label: '资源监控' },
  { key: 'code_quality', label: '代码质量' },
]

const useStyles = makeStyles({
  moduleCheckboxes: {
    display: 'flex',
    flexWrap: 'wrap',
    ...shorthands.gap(tokens.spacingHorizontalM, tokens.spacingVerticalS),
    marginBottom: tokens.spacingVerticalM,
  },
  actionRow: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalM),
    marginTop: tokens.spacingVerticalS,
  },
})

interface AnalysisModuleSelectorProps {
  selectedModules: Set<string>
  onToggleModule: (key: string) => void
  onStartAnalysis: () => void
  onStopAnalysis: () => void
  onToggleHistory: () => void
  analyzing: boolean
}

export function AnalysisModuleSelector({
  selectedModules,
  onToggleModule,
  onStartAnalysis,
  onStopAnalysis,
  onToggleHistory,
  analyzing,
}: AnalysisModuleSelectorProps) {
  const styles = useStyles()

  return (
    <>
      <div className={styles.moduleCheckboxes}>
        {ANALYZER_MODULES.map((mod) => (
          <Checkbox
            key={mod.key}
            label={mod.label}
            checked={selectedModules.has(mod.key)}
            onChange={() => onToggleModule(mod.key)}
            disabled={analyzing}
          />
        ))}
      </div>
      <div className={styles.actionRow}>
        <Button
          appearance="primary"
          icon={<Play24Regular />}
          onClick={onStartAnalysis}
          disabled={analyzing || selectedModules.size === 0}
        >
          开始分析
        </Button>
        {analyzing && (
          <Button appearance="secondary" onClick={onStopAnalysis}>
            停止
          </Button>
        )}
        <Button
          appearance="subtle"
          icon={<History24Regular />}
          onClick={onToggleHistory}
        >
          查看历史
        </Button>
      </div>
    </>
  )
}
