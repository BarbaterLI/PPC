import { createContext, useContext, useCallback, type ReactNode } from 'react'
import {
  Toaster,
  useToastController,
  Toast,
  ToastTitle,
  ToastBody,
  ToastFooter,
  ToastIntent,
  Link,
} from '@fluentui/react-components'

type ToastIntentType = 'success' | 'error' | 'warning' | 'info'

interface ToastOptions {
  title: string
  body?: string
  intent?: ToastIntentType
  action?: {
    label: string
    onClick: () => void
  }
  duration?: number
}

interface ToastContextValue {
  showToast: (options: ToastOptions) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

const intentMap: Record<ToastIntentType, ToastIntent> = {
  success: 'success',
  error: 'error',
  warning: 'warning',
  info: 'info',
}

function ToastController({ children }: { children: ReactNode }) {
  const { dispatchToast } = useToastController()

  const showToast = useCallback(
    (options: ToastOptions) => {
      const { title, body, intent = 'info', action, duration = 5000 } = options

      dispatchToast(
        <Toast>
          <ToastTitle>{title}</ToastTitle>
          {body && <ToastBody>{body}</ToastBody>}
          {action && (
            <ToastFooter>
              <Link onClick={action.onClick}>{action.label}</Link>
            </ToastFooter>
          )}
        </Toast>,
        {
          intent: intentMap[intent],
          timeout: duration,
          position: 'top-end',
        },
      )
    },
    [dispatchToast],
  )

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
    </ToastContext.Provider>
  )
}

export function ToastNotification({ children }: { children: ReactNode }) {
  return (
    <ToastController>
      <Toaster />
      {children}
    </ToastController>
  )
}

export function useToast(): ToastContextValue {
  const context = useContext(ToastContext)
  if (!context) {
    throw new Error('useToast must be used within a ToastNotification provider')
  }
  return context
}