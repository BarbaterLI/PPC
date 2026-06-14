import {
  makeStyles,
  tokens,
  shorthands,
  Text,
  Card,
  CardHeader,
} from '@fluentui/react-components'

const useStyles = makeStyles({
  cards: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
    ...shorthands.gap(tokens.spacingHorizontalM),
    marginBottom: tokens.spacingVerticalL,
  },
  card: {
    ...shorthands.padding(tokens.spacingVerticalM, tokens.spacingHorizontalL),
  },
  cardValue: {
    fontSize: tokens.fontSizeBase600,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
  },
  cardLabel: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
    marginTop: tokens.spacingVerticalXS,
  },
})

interface AnalysisResultCardsProps {
  overallScore: number
  moduleCount: number
  issueCount: number
}

function getScoreColor(score: number) {
  if (score >= 80) return tokens.colorPaletteGreenForeground1
  if (score >= 60) return tokens.colorPaletteYellowForeground1
  return tokens.colorPaletteRedForeground1
}

export function AnalysisResultCards({
  overallScore,
  moduleCount,
  issueCount,
}: AnalysisResultCardsProps) {
  const styles = useStyles()

  const scoreColor = getScoreColor(overallScore)
  const issueColor = issueCount > 0 ? tokens.colorPaletteRedForeground1 : tokens.colorPaletteGreenForeground1

  return (
    <div className={styles.cards}>
      <Card className={styles.card}>
        <CardHeader header={<Text weight="semibold">综合评分</Text>} />
        <div
          className={styles.cardValue}
          style={{ color: scoreColor }}
        >
          {overallScore} / 100
        </div>
        <div className={styles.cardLabel}>
          {overallScore >= 80
            ? '系统运行良好'
            : overallScore >= 60
              ? '存在可优化项'
              : '需要关注'}
        </div>
      </Card>
      <Card className={styles.card}>
        <CardHeader header={<Text weight="semibold">分析模块数</Text>} />
        <div className={styles.cardValue}>{moduleCount}</div>
        <div className={styles.cardLabel}>个模块已完成分析</div>
      </Card>
      <Card className={styles.card}>
        <CardHeader header={<Text weight="semibold">发现问题</Text>} />
        <div
          className={styles.cardValue}
          style={{ color: issueColor }}
        >
          {issueCount}
        </div>
        <div className={styles.cardLabel}>
          {issueCount === 0 ? '未发现问题' : '个问题需要关注'}
        </div>
      </Card>
    </div>
  )
}
