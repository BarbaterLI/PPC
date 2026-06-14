import { type ReactNode } from 'react'
import {
  makeStyles,
  tokens,
  shorthands,
  Text,
} from '@fluentui/react-components'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    ...shorthands.padding(tokens.spacingVerticalL, tokens.spacingHorizontalL),
    ...shorthands.gap(tokens.spacingHorizontalL),
    flexWrap: 'wrap',
  },
  titleGroup: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalXS),
    flex: 1,
    minWidth: 0,
  },
  title: {
    fontWeight: tokens.fontWeightSemibold,
    fontSize: tokens.fontSizeBase500,
    color: tokens.colorNeutralForeground1,
  },
  description: {
    fontSize: tokens.fontSizeBase300,
    color: tokens.colorNeutralForeground2,
    lineHeight: tokens.lineHeightBase300,
  },
  actionArea: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalS),
    flexShrink: 0,
  },
})

interface PageHeaderProps {
  title: string
  description?: string
  actions?: ReactNode
}

export function PageHeader({ title, description, actions }: PageHeaderProps) {
  const styles = useStyles()

  return (
    <div className={styles.root}>
      <div className={styles.titleGroup}>
        <Text as="h1" className={styles.title}>{title}</Text>
        {description && (
          <Text className={styles.description}>{description}</Text>
        )}
      </div>
      {actions && (
        <div className={styles.actionArea}>{actions}</div>
      )}
    </div>
  )
}