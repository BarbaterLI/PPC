import { useState, useCallback, useRef, useEffect, type DragEvent } from 'react'
import { PageHeader } from '@/components/PageHeader'
import { EmptyState } from '@/components/EmptyState'
import { StatusBadge } from '@/components/StatusBadge'
import { ConfirmDialog } from '@/components/ConfirmDialog'
import { useApi } from '@/hooks/useApi'
import { useToast } from '@/components/ToastNotification'
import type { ExtensionInfo } from '@/types'
import {
  makeStyles,
  tokens,
  shorthands,
  mergeClasses,
  Text,
  Button,
  Dialog,
  DialogSurface,
  DialogTitle,
  DialogBody,
  DialogActions,
  DialogContent,
  Drawer,
  DrawerBody,
  DrawerHeader,
  DrawerHeaderTitle,
  Switch,
  Table,
  TableHeader,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  Spinner,
  Tooltip,
  Divider,
} from '@fluentui/react-components'
import {
  PuzzlePiece24Regular,
  PuzzlePiece24Filled,
  Add24Regular,
  Dismiss24Regular,
  ArrowDownload24Regular,
  DocumentFolder24Regular,
  Shield24Regular,
  Person24Regular,
  Tag24Regular,
  Code24Regular,
} from '@fluentui/react-icons'

interface ExtendedExtensionInfo extends ExtensionInfo {
  type?: string
}

const useStyles = makeStyles({
  content: {
    ...shorthands.padding(0, tokens.spacingHorizontalL, tokens.spacingVerticalL),
  },
  tableWrapper: {
    overflowX: 'auto',
  },
  table: {
    width: '100%',
  },
  actionCell: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalXS),
  },
  extensionName: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalS),
  },
  extensionIcon: {
    color: tokens.colorBrandForeground1,
    flexShrink: 0,
  },
  loadingContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: tokens.spacingVerticalXXL,
    paddingBottom: tokens.spacingVerticalXXL,
  },
  errorContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: tokens.spacingVerticalXXL,
    paddingBottom: tokens.spacingVerticalXXL,
    ...shorthands.gap(tokens.spacingVerticalM),
  },
  uploadArea: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    ...shorthands.padding(tokens.spacingVerticalXXL, tokens.spacingHorizontalXXL),
    ...shorthands.border('2px', 'dashed', tokens.colorNeutralStroke2),
    borderRadius: tokens.borderRadiusLarge,
    backgroundColor: tokens.colorNeutralBackground2,
    cursor: 'pointer',
    transitionDuration: tokens.durationNormal,
    transitionTimingFunction: tokens.curveEasyEase,
    transitionProperty: 'border-color, background-color',
    ':hover': {
      ...shorthands.borderColor(tokens.colorBrandForeground1),
      backgroundColor: tokens.colorNeutralBackground3,
    },
  },
  uploadAreaDragOver: {
    ...shorthands.borderColor(tokens.colorBrandForeground1),
    backgroundColor: tokens.colorBrandBackground2,
  },
  uploadIcon: {
    color: tokens.colorNeutralForeground3,
    marginBottom: tokens.spacingVerticalM,
  },
  uploadText: {
    color: tokens.colorNeutralForeground2,
    fontSize: tokens.fontSizeBase300,
  },
  uploadHint: {
    color: tokens.colorNeutralForeground3,
    fontSize: tokens.fontSizeBase200,
    marginTop: tokens.spacingVerticalXS,
  },
  uploadProgress: {
    width: '100%',
    marginTop: tokens.spacingVerticalL,
  },
  progressLabel: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: tokens.spacingVerticalXS,
  },
  detailSection: {
    ...shorthands.padding(tokens.spacingVerticalM, 0),
  },
  detailRow: {
    display: 'flex',
    alignItems: 'flex-start',
    ...shorthands.gap(tokens.spacingHorizontalS),
    marginBottom: tokens.spacingVerticalM,
  },
  detailRowIcon: {
    color: tokens.colorNeutralForeground3,
    flexShrink: 0,
    marginTop: '1px',
  },
  detailLabel: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
    marginBottom: tokens.spacingVerticalXS,
  },
  detailValue: {
    fontSize: tokens.fontSizeBase300,
    color: tokens.colorNeutralForeground1,
  },
  tagList: {
    display: 'flex',
    flexWrap: 'wrap',
    ...shorthands.gap(tokens.spacingHorizontalXS),
  },
  tag: {
    display: 'inline-flex',
    alignItems: 'center',
    height: '24px',
    paddingLeft: tokens.spacingHorizontalS,
    paddingRight: tokens.spacingHorizontalS,
    fontSize: tokens.fontSizeBase200,
    fontWeight: tokens.fontWeightRegular,
    borderRadius: tokens.borderRadiusMedium,
    backgroundColor: tokens.colorNeutralBackground4,
    color: tokens.colorNeutralForeground2,
    whiteSpace: 'nowrap',
  },
  fileItem: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalXS),
    paddingTop: tokens.spacingVerticalXS,
    paddingBottom: tokens.spacingVerticalXS,
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground2,
    fontFamily: 'monospace',
  },
  divider: {
    marginTop: tokens.spacingVerticalM,
    marginBottom: tokens.spacingVerticalM,
  },
  switchRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    ...shorthands.padding(tokens.spacingVerticalM, 0),
  },
  versionCell: {
    fontFamily: 'monospace',
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground2,
  },
})

export default function ExtensionsPage() {
  const styles = useStyles()
  const { showToast } = useToast()

  useEffect(() => { document.title = '扩展管理 - PPC10' }, [])

  const { data: extensions, loading, error, refetch } = useApi<ExtendedExtensionInfo[]>('/api/extensions')

  const [installDialogOpen, setInstallDialogOpen] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [dragOver, setDragOver] = useState(false)
  const [selectedExt, setSelectedExt] = useState<ExtendedExtensionInfo | null>(null)
  const [detailDrawerOpen, setDetailDrawerOpen] = useState(false)
  const [uninstallTarget, setUninstallTarget] = useState<ExtendedExtensionInfo | null>(null)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [togglingExt, setTogglingExt] = useState<string | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const xhrRef = useRef<XMLHttpRequest | null>(null)

  useEffect(() => {
    return () => {
      xhrRef.current?.abort()
    }
  }, [])

  const handleShowDetail = useCallback((ext: ExtendedExtensionInfo) => {
    setSelectedExt(ext)
    setDetailDrawerOpen(true)
  }, [])

  const handleToggleEnabled = useCallback(async (ext: ExtendedExtensionInfo) => {
    const action = ext.enabled ? 'disable' : 'enable'
    setTogglingExt(ext.name)

    try {
      const res = await fetch(`/api/extensions/${encodeURIComponent(ext.name)}/${action}`, {
        method: 'POST',
      })

      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        throw new Error(errText || `操作失败: ${res.status}`)
      }

      showToast({
        title: ext.enabled ? '已禁用' : '已启用',
        body: `扩展 "${ext.name}" 已${ext.enabled ? '禁用' : '启用'}`,
        intent: 'success',
      })

      refetch()
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : '操作失败'
      showToast({
        title: '操作失败',
        body: message,
        intent: 'error',
      })
    } finally {
      setTogglingExt(null)
    }
  }, [refetch, showToast])

  const handleUninstallClick = useCallback((ext: ExtendedExtensionInfo) => {
    setUninstallTarget(ext)
    setConfirmOpen(true)
  }, [])

  const handleUninstallConfirm = useCallback(async () => {
    if (!uninstallTarget) return

    try {
      const res = await fetch(`/api/extensions/${encodeURIComponent(uninstallTarget.name)}`, {
        method: 'DELETE',
      })

      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        throw new Error(errText || `卸载失败: ${res.status}`)
      }

      showToast({
        title: '已卸载',
        body: `扩展 "${uninstallTarget.name}" 已卸载`,
        intent: 'success',
      })

      refetch()
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : '卸载失败'
      showToast({
        title: '卸载失败',
        body: message,
        intent: 'error',
      })
    } finally {
      setUninstallTarget(null)
      setConfirmOpen(false)
    }
  }, [uninstallTarget, refetch, showToast])

  const handleFileSelect = useCallback((file: File) => {
    if (!file.name.endsWith('.ppc10ext.zip')) {
      showToast({
        title: '文件格式错误',
        body: '请选择 .ppc10ext.zip 格式的扩展文件',
        intent: 'warning',
      })
      return
    }

    setUploading(true)
    setUploadProgress(0)

    const xhr = new XMLHttpRequest()
    xhrRef.current = xhr
    const formData = new FormData()
    formData.append('file', file)

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        setUploadProgress(Math.round((e.loaded / e.total) * 100))
      }
    })

    xhr.addEventListener('load', () => {
      setUploading(false)
      setUploadProgress(0)
      setInstallDialogOpen(false)

      if (xhr.status >= 200 && xhr.status < 300) {
        showToast({
          title: '安装成功',
          body: `扩展 "${file.name}" 已安装`,
          intent: 'success',
        })
        refetch()
      } else {
        const errText = xhr.responseText || `安装失败: ${xhr.status}`
        showToast({
          title: '安装失败',
          body: errText,
          intent: 'error',
        })
      }
    })

    xhr.addEventListener('error', () => {
      setUploading(false)
      setUploadProgress(0)
      showToast({
        title: '安装失败',
        body: '网络错误，请重试',
        intent: 'error',
      })
    })

    xhr.open('POST', '/api/extensions/install')
    xhr.send(formData)
  }, [refetch, showToast])

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)

    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }, [handleFileSelect])

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)
  }, [])

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
    if (e.target) {
      e.target.value = ''
    }
  }, [handleFileSelect])

  const extType = (ext: ExtendedExtensionInfo) => ext.type ?? (ext.has_webui ? 'WebUI' : '插件')

  return (
    <div>
      <PageHeader
        title="扩展"
        description="扩展插件管理"
        actions={
          <Button
            appearance="primary"
            icon={<Add24Regular />}
            onClick={() => setInstallDialogOpen(true)}
          >
            安装扩展
          </Button>
        }
      />

      <div className={styles.content}>
        {loading && !extensions ? (
          <div className={styles.loadingContainer}>
            <Spinner label="加载扩展列表..." />
          </div>
        ) : error ? (
          <div className={styles.errorContainer}>
            <Text>加载失败: {error}</Text>
            <Button appearance="secondary" onClick={refetch}>重试</Button>
          </div>
        ) : !extensions || extensions.length === 0 ? (
          <EmptyState
            icon={<PuzzlePiece24Regular fontSize={48} />}
            title="暂无扩展"
            message={'还没有安装任何扩展插件。点击上方「安装扩展」按钮来添加新的扩展。'}
            actionLabel="安装扩展"
            onAction={() => setInstallDialogOpen(true)}
          />
        ) : (
          <div className={styles.tableWrapper}>
            <Table className={styles.table} size="small">
              <TableHeader>
                <TableRow>
                  <TableHeaderCell>名称</TableHeaderCell>
                  <TableHeaderCell>版本</TableHeaderCell>
                  <TableHeaderCell>类型</TableHeaderCell>
                  <TableHeaderCell>状态</TableHeaderCell>
                  <TableHeaderCell>操作</TableHeaderCell>
                </TableRow>
              </TableHeader>
              <TableBody>
                {extensions.map((ext) => (
                  <TableRow key={ext.id}>
                    <TableCell>
                      <div className={styles.extensionName}>
                        <span className={styles.extensionIcon}>
                          {ext.has_webui ? <PuzzlePiece24Filled /> : <PuzzlePiece24Regular />}
                        </span>
                        <div>
                          <Text weight="semibold">{ext.name}</Text>
                          {ext.description && (
                            <Text size={200} style={{ display: 'block', color: tokens.colorNeutralForeground3 }}>
                              {ext.description}
                            </Text>
                          )}
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className={styles.versionCell}>{ext.version}</span>
                    </TableCell>
                    <TableCell>
                      <Text size={200}>{extType(ext)}</Text>
                    </TableCell>
                    <TableCell>
                      {togglingExt === ext.name ? (
                        <Spinner size="extra-tiny" />
                      ) : (
                        <StatusBadge
                          status={ext.enabled ? 'completed' : 'cancelled'}
                          label={ext.enabled ? '已启用' : '已禁用'}
                        />
                      )}
                    </TableCell>
                    <TableCell>
                      <div className={styles.actionCell}>
                        <Tooltip content="查看详情" relationship="label">
                          <Button
                            size="small"
                            appearance="subtle"
                            onClick={() => handleShowDetail(ext)}
                          >
                            详情
                          </Button>
                        </Tooltip>
                        <Tooltip content={ext.enabled ? '禁用' : '启用'} relationship="label">
                          <Button
                            size="small"
                            appearance="subtle"
                            disabled={togglingExt === ext.name}
                            onClick={() => handleToggleEnabled(ext)}
                          >
                            {ext.enabled ? '禁用' : '启用'}
                          </Button>
                        </Tooltip>
                        <Tooltip content="卸载扩展" relationship="label">
                          <Button
                            size="small"
                            appearance="subtle"
                            onClick={() => handleUninstallClick(ext)}
                          >
                            卸载
                          </Button>
                        </Tooltip>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </div>

      <Dialog open={installDialogOpen} onOpenChange={(_, data) => {
        if (!data.open) {
          xhrRef.current?.abort()
          xhrRef.current = null
        }
        setInstallDialogOpen(data.open)
      }}>
        <DialogSurface>
          <DialogBody>
            <DialogTitle>安装扩展</DialogTitle>
            <DialogContent>
              <input
                ref={fileInputRef}
                type="file"
                accept=".ppc10ext.zip"
                style={{ display: 'none' }}
                onChange={handleInputChange}
              />
              <div
                className={mergeClasses(styles.uploadArea, dragOver && styles.uploadAreaDragOver)}
                onClick={() => fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
              >
                {uploading ? (
                  <div className={styles.uploadProgress}>
                    <div className={styles.progressLabel}>
                      <Text size={300}>正在上传...</Text>
                      <Text size={200}>{uploadProgress}%</Text>
                    </div>
                    <progress
                      value={uploadProgress}
                      max={100}
                      style={{
                        width: '100%',
                        height: tokens.spacingVerticalS,
                        borderRadius: tokens.borderRadiusMedium,
                        accentColor: tokens.colorBrandForeground1,
                      }}
                    />
                  </div>
                ) : (
                  <>
                    <ArrowDownload24Regular fontSize={48} className={styles.uploadIcon} />
                    <Text className={styles.uploadText}>拖拽文件到此处，或点击选择文件</Text>
                    <Text className={styles.uploadHint}>仅支持 .ppc10ext.zip 格式</Text>
                  </>
                )}
              </div>
            </DialogContent>
            <DialogActions>
              <Button
                appearance="secondary"
                disabled={uploading}
                onClick={() => setInstallDialogOpen(false)}
              >
                取消
              </Button>
            </DialogActions>
          </DialogBody>
        </DialogSurface>
      </Dialog>

      <Drawer
        open={detailDrawerOpen}
        onOpenChange={(_, { open }) => setDetailDrawerOpen(open)}
        position="end"
        size="medium"
        separator
      >
        <DrawerHeader>
          <DrawerHeaderTitle
            action={
              <Button
                appearance="subtle"
                icon={<Dismiss24Regular />}
                onClick={() => setDetailDrawerOpen(false)}
                aria-label="关闭详情"
              />
            }
          >
            <div className={styles.extensionName}>
              <span className={styles.extensionIcon}>
                {selectedExt?.has_webui ? <PuzzlePiece24Filled /> : <PuzzlePiece24Regular />}
              </span>
              <Text weight="semibold" size={500}>{selectedExt?.name ?? '扩展详情'}</Text>
            </div>
          </DrawerHeaderTitle>
        </DrawerHeader>

        <DrawerBody>
          {selectedExt && (
            <>
              <div className={styles.detailSection}>
                <div className={styles.detailRow}>
                  <Code24Regular className={styles.detailRowIcon} />
                  <div>
                    <div className={styles.detailLabel}>版本</div>
                    <div className={styles.detailValue}>{selectedExt.version}</div>
                  </div>
                </div>

                {selectedExt.description && (
                  <div className={styles.detailRow}>
                    <DocumentFolder24Regular className={styles.detailRowIcon} />
                    <div>
                      <div className={styles.detailLabel}>描述</div>
                      <div className={styles.detailValue}>{selectedExt.description}</div>
                    </div>
                  </div>
                )}

                {selectedExt.author && (
                  <div className={styles.detailRow}>
                    <Person24Regular className={styles.detailRowIcon} />
                    <div>
                      <div className={styles.detailLabel}>作者</div>
                      <div className={styles.detailValue}>{selectedExt.author}</div>
                    </div>
                  </div>
                )}

                <div className={styles.detailRow}>
                  <Shield24Regular className={styles.detailRowIcon} />
                  <div>
                    <div className={styles.detailLabel}>类型</div>
                    <div className={styles.detailValue}>{extType(selectedExt)}</div>
                  </div>
                </div>

                {selectedExt.tags && selectedExt.tags.length > 0 && (
                  <div className={styles.detailRow}>
                    <Tag24Regular className={styles.detailRowIcon} />
                    <div>
                      <div className={styles.detailLabel}>标签</div>
                      <div className={styles.tagList}>
                        {selectedExt.tags.map((tag) => (
                          <span key={tag} className={styles.tag}>{tag}</span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <Divider className={styles.divider} />

              <div className={styles.switchRow}>
                <div>
                  <Text weight="semibold" size={300}>
                    {selectedExt.enabled ? '已启用' : '已禁用'}
                  </Text>
                  <Text size={200} style={{ display: 'block', color: tokens.colorNeutralForeground3 }}>
                    {selectedExt.enabled ? '扩展正在运行' : '扩展已停止'}
                  </Text>
                </div>
                <Switch
                  checked={selectedExt.enabled}
                  disabled={togglingExt === selectedExt.name}
                  onChange={() => handleToggleEnabled(selectedExt)}
                  label={selectedExt.enabled ? '启用中' : '已禁用'}
                  labelPosition="before"
                />
              </div>

              {selectedExt.files && selectedExt.files.length > 0 && (
                <>
                  <Divider className={styles.divider} />
                  <Text weight="semibold" size={300}>文件列表</Text>
                  <div style={{ marginTop: tokens.spacingVerticalS }}>
                    {selectedExt.files.map((file) => (
                      <div key={file} className={styles.fileItem}>
                        <DocumentFolder24Regular fontSize={14} />
                        <span>{file}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </DrawerBody>
      </Drawer>

      <ConfirmDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
        title="确认卸载"
        message={
          <span>
            确定要卸载扩展 <strong>{uninstallTarget?.name}</strong> 吗？此操作无法撤销。
          </span>
        }
        confirmLabel="卸载"
        cancelLabel="取消"
        onConfirm={handleUninstallConfirm}
        onCancel={() => setUninstallTarget(null)}
        danger
      />
    </div>
  )
}