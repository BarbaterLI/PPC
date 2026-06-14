import {
  makeStyles,
  tokens,
  shorthands,
  Text,
  ProgressBar,
} from '@fluentui/react-components'

const useStyles = makeStyles({
  progressArea: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalM),
    ...shorthands.padding(tokens.spacingVerticalM, 0),
  },
  progressBar: {
    width: '100%',
  },
  progressText: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground2,
  },
})

interface AnalysisProgressProps {
  progress: number
  message?: string
}

export function AnalysisProgress({
  progress,
  message,
}: AnalysisProgressProps) {
  const styles = useStyles()

  return (
    <div className={styles.progressArea}>
      <Text weight="semibold">分析进行中...</Text>
      <ProgressBar
        className={styles.progressBar}
        value={progress / 100}
        color="brand"
      />
      <Text className={styles.progressText}>
        {Math.round(progress)}%
        {message && ` — ${message}`}
      </Text>
    </div>
  )
}
