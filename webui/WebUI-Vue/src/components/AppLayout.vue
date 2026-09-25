<script setup lang="ts">
// 应用壳层：品牌栏(系统状态 + 主题切换) + WinNavigationView 导航 + 路由出口。
// 扩展导航项动态注入：从 /api/extensions/webui 拉取扩展 WebUI 配置生成，
// 使扩展页面（如 ext:fanqie）随扩展包加载，而非硬编码在 WebUI 代码中。
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { WinButton, WinNavigationView, WinTextBlock } from 'winui-web-library'
import { useTheme } from '@/theme/useTheme'
import { extensionApi, systemApi } from '@/api/client'
import type { SystemStatus } from '@/api/types'

interface NavEntry {
  Tag: string
  Icon: string
  Content: string
}

const BASE_NAV_ITEMS: NavEntry[] = [
  { Tag: 'dashboard', Icon: '\uE80F', Content: '仪表盘' },
  { Tag: 'tts-convert', Icon: '\uE91F', Content: 'TTS 转换' },
  { Tag: 'config', Icon: '\uE713', Content: '配置' },
  { Tag: 'extensions', Icon: '\uE71D', Content: '扩展' },
  { Tag: 'distributed', Icon: '\uE968', Content: '分布式' },
]

// Fluent 图标名 → Segoe MDL2/Fluent 字体码点 的轻量映射；未知图标回退为默认扩展图标。
const EXT_ICON_FALLBACK = '\uE71D'
const EXT_ICON_GLYPHS: Record<string, string> = {
  Book24Regular: '\uE8F2',
}

const route = useRoute()
const router = useRouter()
const { mode, toggleTheme } = useTheme()

const navItems = ref<NavEntry[]>(BASE_NAV_ITEMS)

const systemStatus = ref<SystemStatus['status']>('unknown')

onMounted(async () => {
  try {
    const res = await systemApi.getStatus()
    systemStatus.value = res.data?.status ?? 'unknown'
  } catch {
    systemStatus.value = 'unknown'
  }
  await loadExtensionNav()
})

const loadExtensionNav = async () => {
  try {
    const res = await extensionApi.getWebUIConfigs()
    const extNavs = (res.data ?? [])
      .filter((c) => typeof c.route === 'string' && c.route)
      .map((c) => ({
        Tag: `ext:${c.extension_name}`,
        Icon: EXT_ICON_GLYPHS[c.icon ?? ''] ?? EXT_ICON_FALLBACK,
        Content: c.title || c.extension_name,
      }))
    if (extNavs.length > 0) {
      navItems.value = [...BASE_NAV_ITEMS, ...extNavs]
    }
  } catch {
    // 扩展 WebUI 配置不可用时保持静态导航
  }
}

const currentTag = computed(() => {
  if (route.name === 'ext') {
    return `ext:${String(route.params.name ?? '')}`
  }
  const name = route.name
  if (typeof name === 'string') return name
  return 'dashboard'
})

const selectedItem = computed<NavEntry | undefined>(() => {
  const entry = navItems.value.find((n) => n.Tag === currentTag.value)
  return entry ? { Tag: entry.Tag, Icon: entry.Icon, Content: entry.Content } : undefined
})

const onItemInvoked = (args: { InvokedItemContainer?: { Tag?: string } }) => {
  const tag = args?.InvokedItemContainer?.Tag
  if (!tag || tag === currentTag.value) return
  const target = tag.startsWith('ext:') ? `/ext/${tag.slice(4)}` : `/${tag}`
  void router.push(target)
}

const statusLabel = computed(() => {
  switch (systemStatus.value) {
    case 'running':
      return '运行中'
    case 'stopped':
      return '已停止'
    case 'error':
      return '错误'
    default:
      return '未知'
  }
})

const statusClass = computed(() => `ppc-status-${systemStatus.value}`)
const isDark = computed(() => mode.value !== 'light')
</script>

<template>
  <div class="ppc-shell">
    <header class="ppc-shell-header">
      <div class="ppc-shell-brand">
        <span class="ppc-shell-brand-dot" aria-hidden="true" />
        <WinTextBlock class="ppc-shell-brand-title" FontSize="18" FontWeight="600" Text="PPC_NEXT" />
      </div>

      <div class="ppc-shell-actions">
        <span class="ppc-status-pill" :class="statusClass" role="status" :aria-label="`系统状态：${statusLabel}`">
          <span class="ppc-status-dot" aria-hidden="true" />
          {{ statusLabel }}
        </span>
        <WinButton
          class="ppc-theme-toggle"
          Style="SubtleButtonStyle"
          :aria-label="isDark ? '切换浅色主题' : '切换深色主题'"
          @Click="toggleTheme">
          <span class="ppc-theme-glyph" aria-hidden="true">{{ isDark ? '\uE706' : '\uE708' }}</span>
        </WinButton>
      </div>
    </header>

    <WinNavigationView
      class="ppc-shell-nav"
      :MenuItems="navItems"
      :SelectedItem="selectedItem"
      :FooterMenuItems="[]"
      IsBackButtonVisible="Collapsed"
      @ItemInvoked="onItemInvoked">
      <main class="ppc-shell-content">
        <router-view v-slot="{ Component }">
          <Transition name="ppc-page" mode="out-in" appear>
            <component :is="Component" />
          </Transition>
        </router-view>
      </main>
    </WinNavigationView>
  </div>
</template>

<style scoped>
.ppc-shell {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}
.ppc-shell-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 48px;
  flex-shrink: 0;
  padding: 0 16px;
  border-bottom: 1px solid var(--ctrl-border-accent, rgba(0, 0, 0, 0.16));
  background-color: var(--layer-default, #f3f3f3);
  z-index: 10;
}
.ppc-shell-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
.ppc-shell-brand-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
  background: linear-gradient(135deg, #0067c0 0%, #0f6cbd 100%);
  box-shadow: 0 0 0 3px rgba(0, 103, 192, 0.18);
}
.ppc-shell-brand-title {
  color: var(--text-primary, #000);
  letter-spacing: 0.02em;
  user-select: none;
  white-space: nowrap;
}
.ppc-shell-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}
/* 系统状态 pill：语义圆点 + 文本，替代原先孤立的圆点。 */
.ppc-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 26px;
  padding: 0 10px;
  border-radius: 13px;
  font-size: 12px;
  font-weight: 600;
  line-height: 1;
  user-select: none;
}
.ppc-status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  display: inline-block;
}
.ppc-status-running {
  color: var(--ppc-status-success, #0f7b0f);
  background-color: var(--ppc-status-success-bg, rgba(15, 123, 15, 0.12));
}
.ppc-status-running .ppc-status-dot {
  background-color: var(--ppc-status-success, #0f7b0f);
  animation: ppc-dot-pulse 1.6s linear infinite;
}
.ppc-status-stopped {
  color: var(--ppc-status-warning, #9d5d00);
  background-color: var(--ppc-status-warning-bg, rgba(157, 93, 0, 0.14));
}
.ppc-status-stopped .ppc-status-dot {
  background-color: var(--ppc-status-warning, #9d5d00);
}
.ppc-status-error {
  color: var(--ppc-status-danger, #c42b1c);
  background-color: var(--ppc-status-danger-bg, rgba(196, 43, 28, 0.12));
}
.ppc-status-error .ppc-status-dot {
  background-color: var(--ppc-status-danger, #c42b1c);
}
.ppc-status-unknown {
  color: var(--text-secondary, #555);
  background-color: rgba(128, 128, 128, 0.1);
}
.ppc-status-unknown .ppc-status-dot {
  background-color: var(--text-disabled, #a0a0a0);
}
@keyframes ppc-dot-pulse {
  0%,
  100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.45;
    transform: scale(0.8);
  }
}
@media (prefers-reduced-motion: reduce) {
  .ppc-status-running .ppc-status-dot {
    animation: none;
  }
}
.ppc-theme-toggle {
  min-width: 36px;
  height: 32px;
}
.ppc-theme-glyph {
  font-family: 'WinUIOnWebIcons', 'Segoe Fluent Icons', 'Segoe MDL2 Assets';
  font-size: 16px;
  line-height: 1;
}
.ppc-shell-nav {
  flex: 1;
  min-height: 0;
}
/* 滚动容器：水平留白交由各页面的 .ppc-page-container 控制，避免双重 padding。 */
.ppc-shell-content {
  overflow: auto;
  min-height: 0;
}

.ppc-page-enter-active,
.ppc-page-leave-active {
  transition:
    opacity 0.18s ease,
    transform 0.18s ease;
}
.ppc-page-enter-from {
  opacity: 0;
  transform: translateY(6px);
}
.ppc-page-leave-to {
  opacity: 0;
}
</style>