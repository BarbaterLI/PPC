import { Suspense, lazy } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { ThemeProvider } from '@/theme/ThemeProvider'
import { AppLayout } from '@/components/AppLayout'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import { ToastNotification } from '@/components/ToastNotification'
import { Spinner } from '@fluentui/react-components'
import { useApi } from '@/hooks/useApi'
import type { ExtensionWebUIConfig, SystemStatus } from '@/types'

const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const TtsConvertPage = lazy(() => import('./pages/TtsConvertPage'))
const ConfigPage = lazy(() => import('./pages/ConfigPage'))
const AnalysisPage = lazy(() => import('./pages/AnalysisPage'))
const ExtensionsPage = lazy(() => import('./pages/ExtensionsPage'))
const FanqiePage = lazy(() => import('./pages/FanqiePage'))
const DistributedPage = lazy(() => import('./pages/DistributedPage'))
const PipelinePage = lazy(() => import('./pages/PipelinePage'))

function PageLoading() {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      minHeight: 200,
    }}>
      <Spinner label="加载中..." />
    </div>
  )
}

function ExtensionIframePage({ config }: { config: ExtensionWebUIConfig }) {
  return (
    <div style={{ width: '100%', height: 'calc(100vh - var(--header-height, 48px))' }}>
      <iframe
        src={config.route}
        style={{ width: '100%', height: '100%', border: 'none' }}
        title={config.title}
        sandbox="allow-scripts allow-forms allow-popups"
      />
    </div>
  )
}

export function App() {
  const { data: extensions, error: extensionsError } = useApi<ExtensionWebUIConfig[]>('/api/extensions/webui')
  const { data: systemStatusData } = useApi<SystemStatus>('/api/status', { refreshInterval: 30000 })
  if (extensionsError) {
    console.warn('Failed to load extensions', extensionsError)
  }

  const systemStatus = systemStatusData?.status ?? 'unknown'

  return (
    <ThemeProvider>
      <BrowserRouter>
        <ErrorBoundary>
          <ToastNotification>
            <Routes>
              <Route element={<AppLayout extensions={extensions ?? []} systemStatus={systemStatus} />}>
                <Route index element={<Suspense fallback={<PageLoading />}><DashboardPage /></Suspense>} />
                <Route path="tts" element={<Suspense fallback={<PageLoading />}><TtsConvertPage /></Suspense>} />
                <Route path="config" element={<Suspense fallback={<PageLoading />}><ConfigPage /></Suspense>} />
                <Route path="analysis" element={<Suspense fallback={<PageLoading />}><AnalysisPage /></Suspense>} />
                <Route path="extensions" element={<Suspense fallback={<PageLoading />}><ExtensionsPage /></Suspense>} />
                <Route path="fanqie" element={<Suspense fallback={<PageLoading />}><FanqiePage /></Suspense>} />
                <Route path="distributed" element={<Suspense fallback={<PageLoading />}><DistributedPage /></Suspense>} />
                <Route path="pipelines" element={<Suspense fallback={<PageLoading />}><PipelinePage /></Suspense>} />
                {(extensions ?? []).map((ext) => (
                  <Route
                    key={ext.extension_name}
                    path={`extension/${ext.extension_name}`}
                    element={<ExtensionIframePage config={ext} />}
                  />
                ))}
              </Route>
            </Routes>
          </ToastNotification>
        </ErrorBoundary>
      </BrowserRouter>
    </ThemeProvider>
  )
}
