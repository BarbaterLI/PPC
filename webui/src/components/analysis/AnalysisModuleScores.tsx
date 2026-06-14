import {
  makeStyles,
  tokens,
  shorthands,
  ProgressBar,
} from '@fluentui/react-components'

const MODULE_LABELS: Record<string, string> = {
  performance: '性能',
  config: '配置',
  errors: '错误',
  dependency: '依赖',
  network: '网络',
  resource: '资源',
  code_quality: '代码质量',
}

const useStyles = makeStyles({
  scoreBarRow: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalM),
    marginBottom: tokens.spacingVerticalS,
  },
  scoreBarLabel: {
    width: '80px',
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground2,
    flexShrink: 0,
  },
  scoreBar: {
    flex: 1,
  },
  scoreBarValue: {
    width: '48px',
    fontSize: tokens.fontSizeBase200,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
    textAlign: 'right',
    flexShrink: 0,
  },
})

interface ModuleScore {
  module: string
  score: number
}

interface AnalysisModuleScoresProps {
  moduleScores: ModuleScore[]
}

function getScoreColor(score: number) {
  if (score >= 80) return tokens.colorPaletteGreenForeground1
  if (score >= 60) return tokens.colorPaletteYellowForeground1
  return tokens.colorPaletteRedForeground1
}

export function AnalysisModuleScores({
  moduleScores,
}: AnalysisModuleScoresProps) {
  const styles = useStyles()

  return (
    <>
      {moduleScores.map((ms) => (
        <div key={ms.module} className={styles.scoreBarRow}>
          <span className={styles.scoreBarLabel}>
            {MODULE_LABELS[ms.module] || ms.module}
          </span>
          <ProgressBar
            className={styles.scoreBar}
            value={ms.score / 100}
            color={ms.score >= 80 ? 'success' : ms.score >= 60 ? 'warning' : 'error'}
          />
          <span
            className={styles.scoreBarValue}
            style={{ color: getScoreColor(ms.score) }}
          >
            {ms.score}
          </span>
        </div>
      ))}
    </>
  )
}
