import {
  makeStyles,
  tokens,
  shorthands,
  mergeClasses,
} from '@fluentui/react-components'

const useStyles = makeStyles({
  badge: {
    display: 'inline-flex',
    alignItems: 'center',
    height: tokens.spacingHorizontalL,
    paddingLeft: tokens.spacingHorizontalS,
    paddingRight: tokens.spacingHorizontalS,
    fontSize: tokens.fontSizeBase200,
    fontWeight: tokens.fontWeightSemibold,
    lineHeight: tokens.lineHeightBase300,
    borderRadius: tokens.borderRadiusMedium,
    whiteSpace: 'nowrap',
    userSelect: 'none',
    ...shorthands.gap(tokens.spacingHorizontalXXS),
  },
  running: {
    backgroundColor: tokens.colorPaletteGreenBackground2,
    color: tokens.colorPaletteGreenForeground2,
  },
  completed: {
    backgroundColor: tokens.colorPaletteGreenBackground2,
    color: tokens.colorPaletteGreenForeground2,
  },
  failed: {
    backgroundColor: tokens.colorPaletteRedBackground2,
    color: tokens.colorPaletteRedForeground2,
  },
  pending: {
    backgroundColor: tokens.colorNeutralBackground4,
    color: tokens.colorNeutralForeground2,
  },
  cancelled: {
    backgroundColor: tokens.colorPaletteYellowBackground2,
    color: tokens.colorPaletteYellowForeground2,
  },
  warning: {
    backgroundColor: tokens.colorPaletteYellowBackground2,
    color: tokens.colorPaletteYellowForeground2,
  },
  dot: {
    width: tokens.spacingHorizontalXXS,
    height: tokens.spacingHorizontalXXS,
    borderRadius: tokens.borderRadiusCircular,
    flexShrink: 0,
  },
  dotRunning: {
    backgroundColor: tokens.colorPaletteGreenForeground1,
  },
  dotCompleted: {
    backgroundColor: tokens.colorPaletteGreenForeground1,
  },
  dotFailed: {
    backgroundColor: tokens.colorPaletteRedForeground1,
  },
  dotPending: {
    backgroundColor: tokens.colorNeutralForeground3,
  },
  dotCancelled: {
    backgroundColor: tokens.colorPaletteYellowForeground1,
  },
  dotWarning: {
    backgroundColor: tokens.colorPaletteYellowForeground1,
  },
})

export type StatusType = 'running' | 'completed' | 'failed' | 'pending' | 'cancelled' | 'warning'

const dotStyleMap: Record<StatusType, string> = {
  running: 'dotRunning',
  completed: 'dotCompleted',
  failed: 'dotFailed',
  pending: 'dotPending',
  cancelled: 'dotCancelled',
  warning: 'dotWarning',
}

const badgeStyleMap: Record<StatusType, string> = {
  running: 'running',
  completed: 'completed',
  failed: 'failed',
  pending: 'pending',
  cancelled: 'cancelled',
  warning: 'warning',
}

const statusLabelMap: Record<StatusType, string> = {
  running: '运行中',
  completed: '已完成',
  failed: '失败',
  pending: '等待中',
  cancelled: '已取消',
  warning: '警告',
}

interface StatusBadgeProps {
  status: StatusType
  label?: string
}

export function StatusBadge({ status, label }: StatusBadgeProps) {
  const styles = useStyles()

  const badgeStyle = styles[badgeStyleMap[status] as keyof typeof styles] ?? styles.badge
  const dotStyle = styles[dotStyleMap[status] as keyof typeof styles] ?? styles.dot

  return (
    <span className={mergeClasses(styles.badge, badgeStyle)}>
      <span className={mergeClasses(styles.dot, dotStyle)} />
      {label ?? statusLabelMap[status]}
    </span>
  )
}