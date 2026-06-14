import { useState } from 'react'
import {
  makeStyles,
  tokens,
  shorthands,
  Text,
  Button,
  Card,
} from '@fluentui/react-components'
import {
  ArrowSync24Regular,
} from '@fluentui/react-icons'

const MOBILE = '@media (max-width: 767px)'

const useStyles = makeStyles({
  resultsCard: {
    ...shorthands.padding(tokens.spacingVerticalL, tokens.spacingHorizontalL),
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalM),
  },
  resultsSummary: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr 1fr',
    ...shorthands.gap(tokens.spacingHorizontalM),
    [MOBILE]: {
      gridTemplateColumns: '1fr 1fr',
    },
  },
  resultStat: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    ...shorthands.padding(tokens.spacingVerticalM),
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusMedium,
    ...shorthands.gap(tokens.spacingVerticalXS),
  },
  resultStatValue: {
    fontSize: tokens.fontSizeBase600,
    fontWeight: tokens.fontWeightSemibold,
  },
  resultStatLabel: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
  },
  successColor: {
    color: tokens.colorPaletteGreenForeground1,
  },
  failureColor: {
    color: tokens.colorPaletteRedForeground1,
  },
  durationColor: {
    color: tokens.colorBrandForeground1,
  },
  failedSection: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalS),
  },
  failedSectionHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    cursor: 'pointer',
    ...shorthands.padding(tokens.spacingVerticalS, 0),
  },
  failedList: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalS),
  },
  failedItem: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.padding(tokens.spacingVerticalS, tokens.spacingHorizontalM),
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusMedium,
    ...shorthands.gap(tokens.spacingVerticalXS),
  },
  failedItemName: {
    fontSize: tokens.fontSizeBase300,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorPaletteRedForeground1,
  },
  failedItemError: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
  },
  actionRow: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalM),
    paddingTop: tokens.spacingVerticalS,
  },
})

interface FailedItem {
  file: string
  error: string
}

interface ConvertResult {
  success_count: number
  failure_count: number
  duration: number
  failed_items: FailedItem[]
}

function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

interface TtsConvertResultProps {
  result: ConvertResult
  onReset: () => void
}

export function TtsConvertResult({
  result,
  onReset,
}: TtsConvertResultProps) {
  const styles = useStyles()
  const [showFailedItems, setShowFailedItems] = useState(false)

  return (
    <Card className={styles.resultsCard}>
      <Text size={400} weight="semibold">转换完成</Text>

      <div className={styles.resultsSummary}>
        <div className={styles.resultStat}>
          <Text size={600} weight="semibold" className={styles.successColor}>
            {result.success_count}
          </Text>
          <Text size={200} className={styles.resultStatLabel}>成功</Text>
        </div>
        <div className={styles.resultStat}>
          <Text size={600} weight="semibold" className={styles.failureColor}>
            {result.failure_count}
          </Text>
          <Text size={200} className={styles.resultStatLabel}>失败</Text>
        </div>
        <div className={styles.resultStat}>
          <Text size={600} weight="semibold" className={styles.durationColor}>
            {formatDuration(result.duration)}
          </Text>
          <Text size={200} className={styles.resultStatLabel}>耗时</Text>
        </div>
      </div>

      {result.failed_items.length > 0 && (
        <div className={styles.failedSection}>
          <div
            className={styles.failedSectionHeader}
            onClick={() => setShowFailedItems((v) => !v)}
          >
            <Text weight="semibold" style={{ color: tokens.colorPaletteRedForeground1 }}>
              失败项目 ({result.failed_items.length})
            </Text>
            <Button
              appearance="transparent"
              size="small"
              icon={<ArrowSync24Regular />}
            >
              {showFailedItems ? '收起' : '展开'}
            </Button>
          </div>

          {showFailedItems && (
            <div className={styles.failedList}>
              {result.failed_items.map((item, idx) => (
                <div key={idx} className={styles.failedItem}>
                  <Text className={styles.failedItemName}>{item.file}</Text>
                  <Text size={200} className={styles.failedItemError}>{item.error}</Text>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className={styles.actionRow}>
        <Button
          appearance="primary"
          icon={<ArrowSync24Regular />}
          onClick={onReset}
        >
          开始新转换
        </Button>
      </div>
    </Card>
  )
}
