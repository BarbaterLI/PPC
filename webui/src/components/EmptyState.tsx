import { type ReactNode } from 'react'
import {
  makeStyles,
  tokens,
  shorthands,
  Text,
  Button,
} from '@fluentui/react-components'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    ...shorthands.padding(tokens.spacingVerticalXXL, tokens.spacingHorizontalL),
    textAlign: 'center',
    minHeight: '200px',
    ...shorthands.gap(tokens.spacingVerticalM),
  },
  iconWrapper: {
    color: tokens.colorNeutralForeground3,
    marginBottom: tokens.spacingVerticalS,
  },
  message: {
    color: tokens.colorNeutralForeground2,
    fontSize: tokens.fontSizeBase300,
    maxWidth: '400px',
  },
  action: {
    marginTop: tokens.spacingVerticalS,
  },
})

interface EmptyStateProps {
  icon?: ReactNode
  title?: string
  message: string
  actionLabel?: string
  onAction?: () => void
}

export function EmptyState({
  icon,
  title,
  message,
  actionLabel,
  onAction,
}: EmptyStateProps) {
  const styles = useStyles()

  return (
    <div className={styles.root}>
      {icon && <div className={styles.iconWrapper}>{icon}</div>}
      {title && (
        <Text weight="semibold" size={500}>{title}</Text>
      )}
      <Text className={styles.message}>{message}</Text>
      {actionLabel && onAction && (
        <div className={styles.action}>
          <Button appearance="primary" onClick={onAction}>
            {actionLabel}
          </Button>
        </div>
      )}
    </div>
  )
}