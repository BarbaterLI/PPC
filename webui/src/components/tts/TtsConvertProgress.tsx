import {
  makeStyles,
  tokens,
  shorthands,
  Text,
  Button,
  ProgressBar,
  Card,
  Spinner,
} from '@fluentui/react-components'
import {
  Dismiss24Regular,
} from '@fluentui/react-icons'

const MOBILE = '@media (max-width: 767px)'

const useStyles = makeStyles({
  progressCard: {
    ...shorthands.padding(tokens.spacingVerticalL, tokens.spacingHorizontalL),
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalM),
  },
  progressHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    ...shorthands.gap(tokens.spacingHorizontalM),
    [MOBILE]: {
      flexDirection: 'column',
      alignItems: 'flex-start',
    },
  },
  progressTitle: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalS),
  },
  progressPercentage: {
    fontSize: tokens.fontSizeBase500,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
  },
  progressBar: {
    width: '100%',
  },
  progressDetails: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr 1fr',
    ...shorthands.gap(tokens.spacingHorizontalM),
    [MOBILE]: {
      gridTemplateColumns: '1fr 1fr',
    },
  },
  progressDetailItem: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalXS),
  },
  progressDetailLabel: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
  },
  progressDetailValue: {
    fontSize: tokens.fontSizeBase300,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
  },
  actionRow: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalM),
    paddingTop: tokens.spacingVerticalS,
  },
})

interface SseProgressData {
  progress: number
  current_file?: string
  speed?: string
  eta?: string
  total_files?: number
  completed_files?: number
}

interface TtsConvertProgressProps {
  progressData: SseProgressData
  onCancel: () => void
}

export function TtsConvertProgress({
  progressData,
  onCancel,
}: TtsConvertProgressProps) {
  const styles = useStyles()

  return (
    <Card className={styles.progressCard}>
      <div className={styles.progressHeader}>
        <div className={styles.progressTitle}>
          <Spinner size="small" />
          <Text size={400} weight="semibold">转换进行中</Text>
        </div>
        <Text className={styles.progressPercentage}>
          {Math.round(progressData.progress)}%
        </Text>
      </div>

      <ProgressBar
        className={styles.progressBar}
        value={progressData.progress / 100}
        color="success"
      />

      <div className={styles.progressDetails}>
        <div className={styles.progressDetailItem}>
          <Text size={200} className={styles.progressDetailLabel}>当前文件</Text>
          <Text size={300} className={styles.progressDetailValue}>
            {progressData.current_file || '-'}
          </Text>
        </div>
        <div className={styles.progressDetailItem}>
          <Text size={200} className={styles.progressDetailLabel}>速度</Text>
          <Text size={300} className={styles.progressDetailValue}>
            {progressData.speed || '-'}
          </Text>
        </div>
        <div className={styles.progressDetailItem}>
          <Text size={200} className={styles.progressDetailLabel}>预计剩余时间</Text>
          <Text size={300} className={styles.progressDetailValue}>
            {progressData.eta || '-'}
          </Text>
        </div>
      </div>

      <div className={styles.actionRow}>
        <Button
          appearance="secondary"
          icon={<Dismiss24Regular />}
          onClick={onCancel}
        >
          取消转换
        </Button>
      </div>
    </Card>
  )
}
