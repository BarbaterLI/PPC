import { useState, useCallback, useEffect, useRef } from 'react'
import { PageHeader } from '@/components/PageHeader'
import { StatusBadge } from '@/components/StatusBadge'
import { ConfirmDialog } from '@/components/ConfirmDialog'
import { useApi } from '@/hooks/useApi'
import { useSSE } from '@/hooks/useSSE'
import { useToast } from '@/components/ToastNotification'
import {
  makeStyles,
  tokens,
  shorthands,
  Text,
  Button,
  Card,
  CardHeader,
  Switch,
  Input,
  Label,
  Spinner,
  Tooltip,
} from '@fluentui/react-components'
import {
  ArrowDownload24Regular,
  ArrowSync24Regular,
  Dismiss24Regular,
  Play24Regular,
  Stop24Regular,
  Open24Regular,
  Globe24Regular,
  Server24Regular,
} from '@fluentui/react-icons'

interface ExtendedFanqieStatus {
  installed: boolean
  version?: string
  latest_version?: string
  update_available: boolean
  exe_path?: string
  server_running: boolean
  server_host?: string
  server_port?: number
  current_version?: string
  available?: boolean
  quota_total?: number
  quota_remaining?: number
  message?: string
}

const useStyles = makeStyles({
  content: {
    ...shorthands.padding(0, tokens.spacingHorizontalL, tokens.spacingVerticalL),
  },
  sections: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
    ...shorthands.gap(tokens.spacingHorizontalL),
    alignItems: 'start',
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalM),
  },
  sectionTitle: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalS),
    paddingBottom: tokens.spacingVerticalS,
  },
  sectionTitleIcon: {
    color: tokens.colorBrandForeground1,
  },
  card: {
    ...shorthands.padding(tokens.spacingVerticalL, tokens.spacingHorizontalL),
  },
  statusGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    ...shorthands.gap(tokens.spacingVerticalS, tokens.spacingHorizontalL),
    marginTop: tokens.spacingVerticalS,
  },
  statusItem: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalXS),
  },
  statusLabel: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
  },
  statusValue: {
    fontSize: tokens.fontSizeBase300,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
  },
  statusValueMono: {
    fontSize: tokens.fontSizeBase300,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground1,
    fontFamily: 'monospace',
  },
  actionBar: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalS),
    flexWrap: 'wrap',
    marginTop: tokens.spacingVerticalS,
  },
  downloadProgress: {
    width: '100%',
  },
  progressBar: {
    width: '100%',
    height: tokens.spacingVerticalS,
    borderRadius: tokens.borderRadiusMedium,
    accentColor: tokens.colorBrandForeground1,
  },
  progressLabel: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: tokens.spacingVerticalXS,
  },
  configRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: tokens.spacingVerticalS,
    paddingBottom: tokens.spacingVerticalS,
  },
  configInput: {
    width: '100%',
  },
  formRow: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalXS),
  },
  formRowInline: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalM),
  },
  serverStatusBar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    ...shorthands.padding(tokens.spacingVerticalM, tokens.spacingHorizontalL),
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusMedium,
    ...shorthands.gap(tokens.spacingHorizontalM),
    flexWrap: 'wrap',
  },
  serverInfo: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalM),
    flexWrap: 'wrap',
  },
  accessLink: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalXS),
    fontSize: tokens.fontSizeBase300,
    fontFamily: 'monospace',
    color: tokens.colorBrandForeground1,
    textDecoration: 'none',
    ':hover': {
      textDecoration: 'underline',
    },
  },
  iframeContainer: {
    width: '100%',
    height: '600px',
    ...shorthands.border('1px', 'solid', tokens.colorNeutralStroke2),
    borderRadius: tokens.borderRadiusMedium,
    overflow: 'hidden',
    marginTop: tokens.spacingVerticalM,
  },
  iframe: {
    width: '100%',
    height: '100%',
    ...shorthands.border('none'),
  },
  iframePlaceholder: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    height: '600px',
    backgroundColor: tokens.colorNeutralBackground2,
    ...shorthands.border('1px', 'solid', tokens.colorNeutralStroke2),
    borderRadius: tokens.borderRadiusMedium,
    ...shorthands.gap(tokens.spacingVerticalM),
    marginTop: tokens.spacingVerticalM,
  },
  updateBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    height: '20px',
    paddingLeft: tokens.spacingHorizontalS,
    paddingRight: tokens.spacingHorizontalS,
    fontSize: tokens.fontSizeBase200,
    fontWeight: tokens.fontWeightSemibold,
    borderRadius: tokens.borderRadiusMedium,
    backgroundColor: tokens.colorPaletteYellowBackground2,
    color: tokens.colorPaletteYellowForeground2,
    whiteSpace: 'nowrap',
    marginLeft: tokens.spacingHorizontalXS,
  },
  loadingContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: tokens.spacingVerticalXXL,
    paddingBottom: tokens.spacingVerticalXXL,
  },
})

export default function FanqiePage() {
  const styles = useStyles()
  const { showToast } = useToast()

  useEffect(() => { document.title = '番茄小说 - PPC10' }, [])

  const { data: statusData, loading, refetch } = useApi<ExtendedFanqieStatus>('/api/fanqie/status', {
    refreshInterval: 15000,
  })

  const [downloading, setDownloading] = useState(false)
  const [downloadProgress, setDownloadProgress] = useState(0)
  const [mirrorEnabled, setMirrorEnabled] = useState(false)
  const [mirrorHost, setMirrorHost] = useState('')
  const [uninstallConfirmOpen, setUninstallConfirmOpen] = useState(false)

  const [serverPort, setServerPort] = useState('18423')
  const [serverPassword, setServerPassword] = useState('')
  const [serverRunning, setServerRunning] = useState(false)
  const [serverStarting, setServerStarting] = useState(false)
  const [serverStopping, setServerStopping] = useState(false)
  const [launchedPort, setLaunchedPort] = useState<number | null>(null)

  const iframeRef = useRef<HTMLIFrameElement>(null)
  const { connect: connectSSE, close: closeSSE } = useSSE()

  useEffect(() => {
    return () => {
      closeSSE()
    }
  }, [closeSSE])

  useEffect(() => {
    if (statusData?.server_running !== undefined) {
      setServerRunning(statusData.server_running)
      if (statusData.server_running && statusData.server_port) {
        setLaunchedPort(statusData.server_port)
      }
    }
  }, [statusData?.server_running, statusData?.server_port])

  const handleInstallUpdate = useCallback(async () => {
    setDownloading(true)
    setDownloadProgress(0)

    try {
      const res = await fetch('/api/fanqie/install', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          use_mirror: mirrorEnabled,
          mirror: mirrorHost || undefined,
          prefer_musl: false,
        }),
      })

      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        throw new Error(errText || `安装失败: ${res.status}`)
      }

      const result = await res.json()
      const taskId = result.task_id

      if (!taskId) {
        showToast({
          title: statusData?.installed ? '更新完成' : '安装完成',
          intent: 'success',
        })
        refetch()
        return
      }

      connectSSE(`/api/tasks/${encodeURIComponent(taskId)}/stream`, {
        progress: (event) => {
          try {
            const data = JSON.parse(event.data)
            if (data.progress !== undefined) {
              setDownloadProgress(Math.round(data.progress))
            }
          } catch (error) {
            console.warn('Failed to parse progress event', error)
          }
        },
        complete: () => {
          closeSSE()
          setDownloadProgress(100)
          showToast({
            title: statusData?.installed ? '更新完成' : '安装完成',
            intent: 'success',
          })
          refetch()
          setTimeout(() => {
            setDownloading(false)
            setDownloadProgress(0)
          }, 1500)
        },
        error: (event) => {
          try {
            const messageEvent = event as MessageEvent
            const data = messageEvent.data ? JSON.parse(messageEvent.data) : {}
            showToast({
              title: '安装失败',
              body: data.message || '安装过程中发生错误',
              intent: 'error',
            })
          } catch (error) {
            console.warn('Failed to parse error event', error)
          }
          closeSSE()
          setDownloading(false)
          setDownloadProgress(0)
        },
      })
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : '安装失败'
      showToast({
        title: '安装失败',
        body: message,
        intent: 'error',
      })
    }
  }, [mirrorEnabled, mirrorHost, statusData, refetch, showToast, connectSSE, closeSSE])

  const handleUninstall = useCallback(async () => {
    try {
      const res = await fetch('/api/fanqie/uninstall', {
        method: 'POST',
      })

      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        throw new Error(errText || `卸载失败: ${res.status}`)
      }

      showToast({
        title: '已卸载',
        body: '番茄小说扩展已卸载',
        intent: 'success',
      })

      setUninstallConfirmOpen(false)
      refetch()
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : '卸载失败'
      showToast({
        title: '卸载失败',
        body: message,
        intent: 'error',
      })
    }
  }, [refetch, showToast])

  const handleLaunchServer = useCallback(async () => {
    setServerStarting(true)

    try {
      const res = await fetch('/api/fanqie/launch-server', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          port: parseInt(serverPort, 10) || 18423,
          password: serverPassword || undefined,
        }),
      })

      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        throw new Error(errText || `启动失败: ${res.status}`)
      }

      const resultPort = parseInt(serverPort, 10) || 18423
      setLaunchedPort(resultPort)
      setServerRunning(true)

      showToast({
        title: '服务器已启动',
        body: `番茄小说 Web 服务器已在端口 ${resultPort} 上运行`,
        intent: 'success',
        action: {
          label: '打开',
          onClick: () => window.open(`http://127.0.0.1:${resultPort}`, '_blank'),
        },
      })
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : '启动失败'
      showToast({
        title: '启动失败',
        body: message,
        intent: 'error',
      })
    } finally {
      setServerStarting(false)
    }
  }, [serverPort, serverPassword, showToast])

  const handleStopServer = useCallback(async () => {
    setServerStopping(true)

    try {
      const res = await fetch('/api/fanqie/stop-server', {
        method: 'POST',
      })

      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        throw new Error(errText || `停止失败: ${res.status}`)
      }

      setServerRunning(false)

      showToast({
        title: '服务器已停止',
        body: '番茄小说 Web 服务器已停止运行',
        intent: 'info',
      })
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : '停止失败'
      showToast({
        title: '停止失败',
        body: message,
        intent: 'error',
      })
    } finally {
      setServerStopping(false)
    }
  }, [showToast])

  const handleOpenInNewTab = useCallback(() => {
    const port = launchedPort ?? (parseInt(serverPort, 10) || 18423)
    window.open(`http://127.0.0.1:${port}`, '_blank')
  }, [launchedPort, serverPort])

  const iframeSrc = `http://127.0.0.1:${launchedPort ?? (parseInt(serverPort, 10) || 18423)}`

  return (
    <div>
      <PageHeader
        title="番茄小说"
        description="番茄小说扩展 - 下载管理与运行控制"
      />

      <div className={styles.content}>
        {loading && !statusData ? (
          <div className={styles.loadingContainer}>
            <Spinner label="加载状态..." />
          </div>
        ) : (
          <div className={styles.sections}>
            <div className={styles.section}>
              <div className={styles.sectionTitle}>
                <ArrowDownload24Regular className={styles.sectionTitleIcon} />
                <Text weight="semibold" size={400}>下载管理</Text>
              </div>

              <Card className={styles.card}>
                <CardHeader
                  header={
                    <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacingHorizontalXS }}>
                      <Text weight="semibold" size={400}>扩展状态</Text>
                      {statusData?.update_available && (
                        <span className={styles.updateBadge}>更新可用</span>
                      )}
                    </div>
                  }
                />

                <div className={styles.statusGrid}>
                  <div className={styles.statusItem}>
                    <span className={styles.statusLabel}>安装状态</span>
                    <span className={styles.statusValue}>
                      {statusData?.installed ? '已安装' : '未安装'}
                    </span>
                  </div>
                  <div className={styles.statusItem}>
                    <span className={styles.statusLabel}>服务可用性</span>
                    <span className={styles.statusValue}>
                      {statusData?.available ? '可用' : '不可用'}
                    </span>
                  </div>
                  <div className={styles.statusItem}>
                    <span className={styles.statusLabel}>当前版本</span>
                    <span className={styles.statusValueMono}>
                      {statusData?.current_version ?? '-'}
                    </span>
                  </div>
                  <div className={styles.statusItem}>
                    <span className={styles.statusLabel}>最新版本</span>
                    <span className={styles.statusValueMono}>
                      {statusData?.latest_version ?? '-'}
                    </span>
                  </div>
                  {statusData?.quota_total != null && (
                    <>
                      <div className={styles.statusItem}>
                        <span className={styles.statusLabel}>配额剩余</span>
                        <span className={styles.statusValue}>
                          {statusData.quota_remaining ?? 0}
                        </span>
                      </div>
                      <div className={styles.statusItem}>
                        <span className={styles.statusLabel}>配额总量</span>
                        <span className={styles.statusValue}>
                          {statusData.quota_total}
                        </span>
                      </div>
                    </>
                  )}
                </div>

                {statusData?.message && (
                  <Text
                    size={200}
                    style={{
                      display: 'block',
                      marginTop: tokens.spacingVerticalM,
                      color: tokens.colorNeutralForeground3,
                    }}
                  >
                    {statusData.message}
                  </Text>
                )}

                <div className={styles.actionBar}>
                  <Button
                    appearance="primary"
                    icon={statusData?.update_available ? <ArrowSync24Regular /> : <ArrowDownload24Regular />}
                    disabled={downloading}
                    onClick={handleInstallUpdate}
                  >
                    {downloading
                      ? '下载中...'
                      : statusData?.installed
                        ? (statusData?.update_available ? '更新' : '重新安装')
                        : '安装'}
                  </Button>

                  {statusData?.installed && (
                    <Button
                      appearance="subtle"
                      icon={<Dismiss24Regular />}
                      onClick={() => setUninstallConfirmOpen(true)}
                    >
                      卸载
                    </Button>
                  )}
                </div>

                {downloading && (
                  <div className={styles.downloadProgress} style={{ marginTop: tokens.spacingVerticalM }}>
                    <div className={styles.progressLabel}>
                      <Text size={200}>下载进度</Text>
                      <Text size={200}>{downloadProgress}%</Text>
                    </div>
                    <progress
                      value={downloadProgress}
                      max={100}
                      className={styles.progressBar}
                    />
                  </div>
                )}
              </Card>

              <Card className={styles.card}>
                <CardHeader
                  header={<Text weight="semibold" size={400}>下载配置</Text>}
                />

                <div className={styles.configRow}>
                  <Label htmlFor="mirror-toggle">使用镜像源</Label>
                  <Switch
                    id="mirror-toggle"
                    checked={mirrorEnabled}
                    onChange={(_, data) => setMirrorEnabled(data.checked)}
                    label={mirrorEnabled ? '已启用' : '已禁用'}
                    labelPosition="before"
                  />
                </div>

                {mirrorEnabled && (
                  <div className={styles.configInput} style={{ marginTop: tokens.spacingVerticalS }}>
                    <Input
                      placeholder="输入镜像源地址"
                      value={mirrorHost}
                      onChange={(_, data) => setMirrorHost(data.value)}
                      contentBefore={<Globe24Regular />}
                    />
                  </div>
                )}
              </Card>
            </div>

            <div className={styles.section}>
              <div className={styles.sectionTitle}>
                <Server24Regular className={styles.sectionTitleIcon} />
                <Text weight="semibold" size={400}>使用运行</Text>
              </div>

              <Card className={styles.card}>
                <CardHeader
                  header={<Text weight="semibold" size={400}>服务器配置</Text>}
                />

                <div className={styles.formRow}>
                  <Label htmlFor="server-port">端口号</Label>
                  <Input
                    id="server-port"
                    type="number"
                    value={serverPort}
                    onChange={(_, data) => setServerPort(data.value)}
                    placeholder="18423"
                    disabled={serverRunning}
                    contentBefore={<Globe24Regular />}
                  />
                </div>

                <div className={styles.formRow} style={{ marginTop: tokens.spacingVerticalM }}>
                  <Label htmlFor="server-password">访问密码（可选）</Label>
                  <Input
                    id="server-password"
                    type="password"
                    value={serverPassword}
                    onChange={(_, data) => setServerPassword(data.value)}
                    placeholder="留空则不设置密码"
                    disabled={serverRunning}
                  />
                </div>

                <div className={styles.actionBar} style={{ marginTop: tokens.spacingVerticalM }}>
                  {!serverRunning ? (
                    <Button
                      appearance="primary"
                      icon={<Play24Regular />}
                      disabled={serverStarting}
                      onClick={handleLaunchServer}
                    >
                      {serverStarting ? '启动中...' : '启动 Web 服务器'}
                    </Button>
                  ) : (
                    <Button
                      appearance="primary"
                      icon={<Stop24Regular />}
                      disabled={serverStopping}
                      onClick={handleStopServer}
                    >
                      {serverStopping ? '停止中...' : '停止服务器'}
                    </Button>
                  )}
                </div>
              </Card>

              {serverRunning && (
                <>
                  <div className={styles.serverStatusBar}>
                    <div className={styles.serverInfo}>
                      <StatusBadge status="running" label="运行中" />
                      <a
                        className={styles.accessLink}
                        href={iframeSrc}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <Globe24Regular fontSize={16} />
                        {iframeSrc}
                      </a>
                    </div>
                    <Tooltip content="在新标签页中打开" relationship="label">
                      <Button
                        appearance="subtle"
                        icon={<Open24Regular />}
                        onClick={handleOpenInNewTab}
                      >
                        在新窗口打开
                      </Button>
                    </Tooltip>
                  </div>

                  <div className={styles.iframeContainer}>
                    <iframe
                      ref={iframeRef}
                      src={iframeSrc}
                      className={styles.iframe}
                      title="番茄小说 WebUI"
                      sandbox="allow-scripts allow-forms allow-popups"
                    />
                  </div>
                </>
              )}

              {!serverRunning && (
                <div className={styles.iframePlaceholder}>
                  <Server24Regular fontSize={48} style={{ color: tokens.colorNeutralForeground3 }} />
                  <Text size={400} style={{ color: tokens.colorNeutralForeground3 }}>
                    服务器未启动
                  </Text>
                  <Text size={200} style={{ color: tokens.colorNeutralForeground4 }}>
                    启动服务器后可在此处嵌入番茄小说 WebUI
                  </Text>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <ConfirmDialog
        open={uninstallConfirmOpen}
        onOpenChange={setUninstallConfirmOpen}
        title="确认卸载"
        message="确定要卸载番茄小说扩展吗？此操作无法撤销。"
        confirmLabel="卸载"
        cancelLabel="取消"
        onConfirm={handleUninstall}
        onCancel={() => setUninstallConfirmOpen(false)}
        danger
      />
    </div>
  )
}