import { useState, useMemo, useRef, useCallback, useEffect } from 'react'
import { PageHeader } from '@/components/PageHeader'
import { useApi } from '@/hooks/useApi'
import { useToast } from '@/components/ToastNotification'
import { configApi } from '@/api/client'
import type { PPC10Config } from '@/types'
import {
  makeStyles,
  tokens,
  shorthands,
  Text,
  Button,
  TabList,
  Tab,
  Spinner,
  Badge,
} from '@fluentui/react-components'
import {
  Save24Regular,
  ArrowDownload24Regular,
  ArrowUpload24Regular,
  ArrowReset24Regular,
  Wand24Regular,
} from '@fluentui/react-icons'

import { ConfigField } from '@/components/config/ConfigField'
import { ConfigWizard } from '@/components/config/ConfigWizard'
import { ConfigResetDialog } from '@/components/config/ConfigResetDialog'

const CATEGORY_TABS: Record<string, string> = {
  core: '核心',
  tts: 'TTS',
  split: '分割',
  performance: '性能',
  network: '网络',
  reliability: '可靠性',
  extension: '扩展',
  output: '输出',
}

const TAB_ORDER = ['core', 'tts', 'split', 'performance', 'network', 'reliability', 'extension', 'output']

const useStyles = makeStyles({
  content: {
    ...shorthands.padding(0, tokens.spacingHorizontalL, tokens.spacingVerticalL),
  },
  toolbar: {
    ...shorthands.padding(tokens.spacingVerticalS, 0),
    marginBottom: tokens.spacingVerticalM,
  },
  tabContent: {
    display: 'flex',
    flexDirection: 'column',
    ...shorthands.gap(tokens.spacingVerticalL),
  },
  loadingContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '200px',
  },
  changedBadge: {
    marginLeft: tokens.spacingHorizontalS,
  },
  saveButton: {
    marginTop: tokens.spacingVerticalL,
  },
})

type ConfigMap = Record<string, string>

export default function ConfigPage() {
  const styles = useStyles()
  const { showToast } = useToast()

  useEffect(() => { document.title = '配置管理 - PPC10' }, [])

  const { data: configs, loading, refetch } = useApi<PPC10Config[]>(() => configApi.getAll())
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [editedValues, setEditedValues] = useState<ConfigMap>({})
  const [saving, setSaving] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [importing, setImporting] = useState(false)
  const [resetting, setResetting] = useState(false)
  const [resetDialogOpen, setResetDialogOpen] = useState(false)
  const [selectedTab, setSelectedTab] = useState<string>(localStorage.getItem('config-selected-tab') || '')
  const [wizardOpen, setWizardOpen] = useState(false)

  useEffect(() => { localStorage.setItem('config-selected-tab', selectedTab) }, [selectedTab])

  const groupedConfigs = useMemo(() => {
    if (!configs) return {}
    const groups: Record<string, PPC10Config[]> = {}
    for (const cfg of configs) {
      const cat = cfg.category || 'core'
      if (!groups[cat]) groups[cat] = []
      groups[cat].push(cfg)
    }
    return groups
  }, [configs])

  const originalValues = useMemo(() => {
    if (!configs) return {} as ConfigMap
    const map: ConfigMap = {}
    for (const cfg of configs) {
      map[cfg.key] = cfg.value ?? cfg.default_value ?? ''
    }
    return map
  }, [configs])

  const changedCount = useMemo(() => {
    return Object.entries(editedValues).filter(
      ([key, value]) => value !== (originalValues[key] ?? ''),
    ).length
  }, [editedValues, originalValues])

  const getValue = useCallback(
    (cfg: PPC10Config): string => {
      if (cfg.key in editedValues) return editedValues[cfg.key] ?? ''
      return cfg.value ?? cfg.default_value ?? ''
    },
    [editedValues],
  )

  const isChanged = useCallback(
    (cfg: PPC10Config): boolean => {
      if (!(cfg.key in editedValues)) return false
      const original = cfg.value ?? cfg.default_value ?? ''
      return editedValues[cfg.key] !== original
    },
    [editedValues],
  )

  const handleValueChange = useCallback(
    (key: string, value: string) => {
      setEditedValues((prev) => ({ ...prev, [key]: value }))
    },
    [],
  )

  const handleSave = useCallback(async () => {
    if (!configs || changedCount === 0) return
    setSaving(true)
    try {
      const payloads = Object.entries(editedValues).map(([key, value]) => {
        const cfg = configs.find((c) => c.key === key)
        if (!cfg) return null
        const original = cfg.value ?? cfg.default_value ?? ''
        if (value === original) return null
        return { key, value }
      }).filter(Boolean) as Array<{ key: string; value: string }>

      if (payloads.length === 0) {
        showToast({ title: '没有需要保存的更改', intent: 'info' })
        return
      }

      const result = await configApi.batchUpdate(payloads)
      if (result.success === false) {
        throw new Error(result.error || result.message || '批量保存失败')
      }
      const batchData = result.data
      if (batchData?.failed && batchData.failed.length > 0) {
        const failedKeys = batchData.failed.map((f) => f.key).join(', ')
        showToast({ title: '部分保存失败', body: `失败项: ${failedKeys}`, intent: 'warning' })
      } else {
        showToast({ title: '配置已保存', body: `成功保存 ${payloads.length} 项配置`, intent: 'success' })
      }
      setEditedValues({})
      refetch()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '保存失败'
      showToast({ title: '保存失败', body: msg, intent: 'error' })
    } finally {
      setSaving(false)
    }
  }, [configs, editedValues, changedCount, showToast, refetch])

  const handleExport = useCallback(async () => {
    setExporting(true)
    try {
      const blob = await configApi.exportFile()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `ppc10-config-${new Date().toISOString().slice(0, 10)}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      showToast({ title: '导出成功', body: '配置文件已下载', intent: 'success' })
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '导出失败'
      showToast({ title: '导出失败', body: msg, intent: 'error' })
    } finally {
      setExporting(false)
    }
  }, [showToast])

  const handleImportClick = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  const handleImportFile = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (!file) return
      setImporting(true)
      try {
        await configApi.importFile(file)
        showToast({ title: '导入成功', body: `已从 ${file.name} 导入配置`, intent: 'success' })
        setEditedValues({})
        refetch()
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : '导入失败'
        showToast({ title: '导入失败', body: msg, intent: 'error' })
      } finally {
        setImporting(false)
        if (fileInputRef.current) {
          fileInputRef.current.value = ''
        }
      }
    },
    [showToast, refetch],
  )

  const handleReset = useCallback(async (preset: string) => {
    setResetting(true)
    try {
      const result = await configApi.reset(preset)
      if (result.success === false) {
        throw new Error(result.error || result.message || '重置失败')
      }
      showToast({ title: '重置成功', body: `已使用预设 ${preset} 重置配置`, intent: 'success' })
      setEditedValues({})
      setResetDialogOpen(false)
      refetch()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '重置失败'
      showToast({ title: '重置失败', body: msg, intent: 'error' })
    } finally {
      setResetting(false)
    }
  }, [showToast, refetch])

  const handleWizardComplete = useCallback((data: { voice: string; concurrency: number; outputFormat: string }) => {
    setEditedValues((prev) => ({
      ...prev,
      'tts.default_voice': data.voice,
      'tts.concurrency': String(data.concurrency),
      'output.format': data.outputFormat,
    }))
    setWizardOpen(false)
    showToast({ title: '向导完成', body: '核心配置已更新，请点击保存以应用更改', intent: 'info' })
  }, [showToast])

  const renderTabContent = useCallback((category: string) => {
    const items = groupedConfigs[category]
    if (!items || items.length === 0) {
      return <Text>此分类下没有配置项</Text>
    }
    return (
      <div className={styles.tabContent}>
        {items.map((cfg) => (
          <ConfigField
            key={cfg.key}
            config={cfg}
            value={getValue(cfg)}
            isChanged={isChanged(cfg)}
            onChange={(value) => handleValueChange(cfg.key, value)}
          />
        ))}
      </div>
    )
  }, [groupedConfigs, getValue, isChanged, handleValueChange])

  const availableTabs = TAB_ORDER.filter((cat) => groupedConfigs[cat] && groupedConfigs[cat].length > 0)

  if (loading && !configs) {
    return (
      <div>
        <PageHeader title="配置" description="系统配置管理" />
        <div className={styles.loadingContainer}>
          <Spinner label="加载配置中..." />
        </div>
      </div>
    )
  }

  return (
    <div>
      <PageHeader
        title="配置"
        description="管理系统核心配置参数"
        actions={
          <div style={{ display: 'flex', gap: tokens.spacingHorizontalS }}>
            <Button
              appearance="primary"
              icon={<Save24Regular />}
              onClick={handleSave}
              disabled={changedCount === 0 || saving}
            >
              保存
              {changedCount > 0 && (
                <Badge className={styles.changedBadge} appearance="filled" color="warning" size="small">
                  {changedCount}
                </Badge>
              )}
            </Button>
          </div>
        }
      />

      <div className={styles.content}>
        <div className={styles.toolbar}>
          <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacingHorizontalS, flexWrap: 'wrap' }}>
            <Button appearance="subtle" icon={<ArrowUpload24Regular />} onClick={handleImportClick} disabled={importing}>
              导入
            </Button>
            <Button appearance="subtle" icon={<ArrowDownload24Regular />} onClick={handleExport} disabled={exporting}>
              导出
            </Button>
            <Button appearance="subtle" icon={<ArrowReset24Regular />} onClick={() => setResetDialogOpen(true)} disabled={resetting}>
              重置
            </Button>
            <Button appearance="subtle" icon={<Wand24Regular />} onClick={() => setWizardOpen(true)}>
              配置向导
            </Button>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            style={{ display: 'none' }}
            onChange={handleImportFile}
          />
        </div>

        <TabList
          selectedValue={selectedTab || availableTabs[0]}
          onTabSelect={(_, data) => setSelectedTab(data.value as string)}
        >
          {availableTabs.map((cat) => (
            <Tab key={cat} value={cat}>
              {CATEGORY_TABS[cat] || cat}
            </Tab>
          ))}
        </TabList>

        <div style={{ marginTop: tokens.spacingVerticalL }}>
          {renderTabContent(selectedTab || availableTabs[0] || 'core')}
        </div>

        <div className={styles.saveButton}>
          <Button
            appearance="primary"
            icon={<Save24Regular />}
            onClick={handleSave}
            disabled={changedCount === 0 || saving}
            size="large"
          >
            保存所有更改
            {changedCount > 0 && (
              <Badge className={styles.changedBadge} appearance="filled" color="warning" size="small">
                {changedCount}
              </Badge>
            )}
          </Button>
        </div>
      </div>

      <ConfigResetDialog
        open={resetDialogOpen}
        onClose={() => setResetDialogOpen(false)}
        onReset={handleReset}
        resetting={resetting}
      />

      <ConfigWizard
        open={wizardOpen}
        onClose={() => setWizardOpen(false)}
        onComplete={handleWizardComplete}
      />
    </div>
  )
}
