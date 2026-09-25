/**
 * useExtensionActions —— 复刻 React 工程 `src/components/extensions/useExtensionActions.ts`。
 * 启停 / 卸载：统一经 useApiAction 执行并弹 toast，成功后回调 refetch。
 * 卸载的二次确认由调用方（页面）经共享 ConfirmDialog 完成后再触发动作。
 */
import { ref } from 'vue'
import { useApiAction } from '@/composables/useApiAction'
import { extensionApi } from '@/api/client'
import type { Extension } from '@/api/types'

export function useExtensionActions(refetch: () => void) {
  const togglingExt = ref<string | null>(null)

  let toggleTarget: Extension | null = null

  const { run: runToggle } = useApiAction(
    async () => {
      const ext = toggleTarget
      if (!ext) return
      const result = ext.enabled
        ? await extensionApi.disable(ext.name)
        : await extensionApi.enable(ext.name)
      if (result.success === false) {
        throw new Error(result.error?.message || '操作失败')
      }
    },
    {
      successToast: () => {
        const ext = toggleTarget
        const enabled = ext?.enabled
        return {
          title: enabled ? '已禁用' : '已启用',
          body: `扩展 "${ext?.name}" 已${enabled ? '禁用' : '启用'}`,
          intent: 'success',
        }
      },
      onSuccess: refetch,
    },
  )

  const handleToggleEnabled = async (ext: Extension): Promise<void> => {
    toggleTarget = ext
    togglingExt.value = ext.name
    try {
      await runToggle()
    } finally {
      togglingExt.value = null
      toggleTarget = null
    }
  }

  let uninstallTarget: Extension | null = null

  const { run: runUninstall, loading: uninstalling } = useApiAction(
    async () => {
      const ext = uninstallTarget
      if (!ext) return
      const result = await extensionApi.uninstall(ext.name)
      if (result.success === false) {
        throw new Error(result.error?.message || '卸载失败')
      }
    },
    {
      successToast: () => ({
        title: '已卸载',
        body: `扩展 "${uninstallTarget?.name}" 已卸载`,
        intent: 'success',
      }),
      onSuccess: refetch,
    },
  )

  const uninstallExtension = (ext: Extension): Promise<void> => {
    uninstallTarget = ext
    return runUninstall()
  }

  return {
    togglingExt,
    uninstalling,
    handleToggleEnabled,
    uninstallExtension,
  }
}