import { Component, type ErrorInfo, type ReactNode } from 'react'
import {
  makeStyles,
  tokens,
  shorthands,
  Button,
  Text,
} from '@fluentui/react-components'
import { ErrorCircle24Regular } from '@fluentui/react-icons'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    ...shorthands.padding(tokens.spacingVerticalXXL, tokens.spacingHorizontalL),
    textAlign: 'center',
    minHeight: '300px',
    ...shorthands.gap(tokens.spacingVerticalL),
  },
  icon: {
    color: tokens.colorPaletteRedForeground1,
    fontSize: tokens.fontSizeHero800,
  },
  title: {
    color: tokens.colorNeutralForeground1,
    fontWeight: tokens.fontWeightSemibold,
    fontSize: tokens.fontSizeBase500,
  },
  message: {
    color: tokens.colorNeutralForeground2,
    fontSize: tokens.fontSizeBase300,
    maxWidth: '480px',
  },
  details: {
    color: tokens.colorNeutralForeground3,
    fontSize: tokens.fontSizeBase200,
    fontFamily: 'Consolas, "Courier New", monospace',
    maxWidth: '480px',
    wordBreak: 'break-all',
    backgroundColor: tokens.colorNeutralBackground2,
    ...shorthands.padding(tokens.spacingVerticalS, tokens.spacingHorizontalM),
    borderRadius: tokens.borderRadiusMedium,
    textAlign: 'left',
    overflow: 'auto',
  },
})

interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: ReactNode
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

export function ErrorBoundaryContent({ error, onReset }: { error: Error; onReset: () => void }) {
  const styles = useStyles()

  return (
    <div className={styles.root} role="alert">
      <ErrorCircle24Regular className={styles.icon} />
      <Text className={styles.title}>出现错误</Text>
      <Text className={styles.message}>
        应用遇到了意外错误。请尝试刷新页面或联系管理员。
      </Text>
      {error.message && (
        <pre className={styles.details}>
          {error.message}
        </pre>
      )}
      <Button appearance="primary" onClick={onReset}>
        重试
      </Button>
    </div>
  )
}

class ErrorBoundaryClass extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <ErrorBoundaryContent
          error={this.state.error ?? new Error('Unknown error')}
          onReset={this.handleReset}
        />
      )
    }

    return this.props.children
  }
}

export { ErrorBoundaryClass as ErrorBoundary }