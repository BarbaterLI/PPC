import { useState } from 'react'
import { Outlet, useLocation, Link } from 'react-router-dom'
import {
  makeStyles,
  tokens,
  shorthands,
  Button,
  Drawer,
  DrawerBody,
  DrawerHeader,
  DrawerHeaderTitle,
  Text,
  Tooltip,
  mergeClasses,
} from '@fluentui/react-components'
import {
  Home24Regular,
  Mic24Regular,
  Settings24Regular,
  DataTrending24Regular,
  PuzzlePiece24Regular,
  WeatherSunny24Regular,
  WeatherMoon24Regular,
  Navigation24Regular,
  Dismiss24Regular,
  BookOpen24Regular,
  Server24Regular,
  PanelLeftContract24Regular,
  PanelLeftExpand24Regular,
  Pipeline24Regular,
} from '@fluentui/react-icons'
import { useThemeStore } from '@/theme/ThemeProvider'
import type { ExtensionWebUIConfig } from '@/types'

const SIDEBAR_WIDTH = '240px'
const HEADER_HEIGHT = '48px'
const MOBILE_BREAKPOINT = '@media (max-width: 767px)'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    minHeight: '100vh',
    backgroundColor: tokens.colorNeutralBackground1,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: HEADER_HEIGHT,
    paddingLeft: tokens.spacingHorizontalL,
    paddingRight: tokens.spacingHorizontalL,
    backgroundColor: tokens.colorNeutralBackground1,
    borderBottom: `1px solid ${tokens.colorNeutralStroke2}`,
    position: 'sticky',
    top: 0,
    zIndex: 100,
    flexShrink: 0,
    ...shorthands.gap(tokens.spacingHorizontalM),
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalS),
  },
  headerLogo: {
    fontWeight: tokens.fontWeightSemibold,
    fontSize: tokens.fontSizeBase400,
    color: tokens.colorNeutralForeground1,
    userSelect: 'none',
  },
  headerRight: {
    display: 'flex',
    alignItems: 'center',
    ...shorthands.gap(tokens.spacingHorizontalS),
  },
  statusDot: {
    width: tokens.spacingHorizontalXS,
    height: tokens.spacingHorizontalXS,
    borderRadius: tokens.borderRadiusCircular,
    flexShrink: 0,
  },
  statusRunning: {
    backgroundColor: tokens.colorPaletteGreenForeground1,
  },
  statusWarning: {
    backgroundColor: tokens.colorPaletteYellowForeground1,
  },
  statusError: {
    backgroundColor: tokens.colorPaletteRedForeground1,
  },
  statusUnknown: {
    backgroundColor: tokens.colorNeutralStroke2,
  },
  menuButton: {
    display: 'none',
    [MOBILE_BREAKPOINT]: {
      display: 'flex',
    },
  },
  sidebarToggle: {
    display: 'flex',
    [MOBILE_BREAKPOINT]: {
      display: 'none',
    },
  },
  body: {
    display: 'flex',
    flex: 1,
    minHeight: 0,
  },
  sidebar: {
    width: SIDEBAR_WIDTH,
    flexShrink: 0,
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: tokens.colorNeutralBackground2,
    borderRight: `1px solid ${tokens.colorNeutralStroke2}`,
    overflowY: 'auto',
    overflowX: 'hidden',
    paddingTop: tokens.spacingVerticalS,
    paddingBottom: tokens.spacingVerticalS,
    transition: 'width 200ms ease',
    [MOBILE_BREAKPOINT]: {
      display: 'none',
    },
  },
  sidebarCollapsed: {
    width: tokens.spacingHorizontalXXXL,
    [MOBILE_BREAKPOINT]: {
      display: 'none',
    },
  },
  navItemCollapsed: {
    justifyContent: 'center',
    paddingLeft: 0,
    paddingRight: 0,
    marginLeft: 0,
    marginRight: 0,
  },
  sidebarSection: {
    paddingTop: tokens.spacingVerticalS,
    paddingBottom: tokens.spacingVerticalS,
  },
  sidebarSectionLabel: {
    paddingLeft: tokens.spacingHorizontalL,
    paddingRight: tokens.spacingHorizontalL,
    paddingTop: tokens.spacingVerticalS,
    paddingBottom: tokens.spacingVerticalXS,
    fontSize: tokens.fontSizeBase200,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground3,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  navItem: {
    display: 'flex',
    alignItems: 'center',
    height: tokens.spacingHorizontalXXL,
    paddingLeft: tokens.spacingHorizontalL,
    paddingRight: tokens.spacingHorizontalL,
    ...shorthands.gap(tokens.spacingHorizontalS),
    color: tokens.colorNeutralForeground2,
    backgroundColor: 'transparent',
    ...shorthands.borderRadius(tokens.borderRadiusMedium),
    ...shorthands.margin('0', tokens.spacingHorizontalS),
    cursor: 'pointer',
    textDecoration: 'none',
    fontSize: tokens.fontSizeBase300,
    fontWeight: tokens.fontWeightRegular,
    transition: 'background-color 100ms ease, color 100ms ease',
    ':hover': {
      backgroundColor: tokens.colorNeutralBackground3,
      color: tokens.colorNeutralForeground1,
      textDecoration: 'none',
    },
  },
  navItemActive: {
    backgroundColor: tokens.colorNeutralBackground4,
    color: tokens.colorNeutralForeground1,
    fontWeight: tokens.fontWeightSemibold,
    ':hover': {
      backgroundColor: tokens.colorNeutralBackground4,
    },
  },
  navItemIcon: {
    flexShrink: 0,
    width: tokens.spacingHorizontalL,
    height: tokens.spacingHorizontalL,
  },
  navItemLabel: {
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  main: {
    flex: 1,
    minWidth: 0,
    overflowY: 'auto',
    animationName: {
      from: { opacity: 0 },
      to: { opacity: 1 },
    },
    animationDuration: tokens.durationNormal,
    animationTimingFunction: tokens.curveEasyEase,
    paddingLeft: tokens.spacingHorizontalL,
    paddingRight: tokens.spacingHorizontalL,
    paddingTop: tokens.spacingVerticalL,
    paddingBottom: tokens.spacingVerticalL,
  },
  drawerNavItem: {
    display: 'flex',
    alignItems: 'center',
    height: tokens.spacingHorizontalXXXL,
    paddingLeft: tokens.spacingHorizontalL,
    paddingRight: tokens.spacingHorizontalL,
    ...shorthands.gap(tokens.spacingHorizontalM),
    color: tokens.colorNeutralForeground2,
    backgroundColor: 'transparent',
    ...shorthands.borderRadius(tokens.borderRadiusMedium),
    ...shorthands.margin('0', tokens.spacingHorizontalS),
    cursor: 'pointer',
    textDecoration: 'none',
    fontSize: tokens.fontSizeBase300,
    ':hover': {
      backgroundColor: tokens.colorNeutralBackground3,
      color: tokens.colorNeutralForeground1,
      textDecoration: 'none',
    },
  },
  drawerNavItemActive: {
    backgroundColor: tokens.colorNeutralBackground4,
    color: tokens.colorNeutralForeground1,
    fontWeight: tokens.fontWeightSemibold,
  },
  extensionIcon: {
    flexShrink: 0,
    width: tokens.spacingHorizontalL,
    height: tokens.spacingHorizontalL,
  },
})

interface NavItem {
  path: string
  label: string
  icon: React.ReactElement
}

const builtInNavItems: NavItem[] = [
  { path: '/', label: '仪表盘', icon: <Home24Regular /> },
  { path: '/tts', label: 'TTS 转换', icon: <Mic24Regular /> },
  { path: '/config', label: '配置', icon: <Settings24Regular /> },
  { path: '/analysis', label: '分析', icon: <DataTrending24Regular /> },
  { path: '/extensions', label: '扩展', icon: <PuzzlePiece24Regular /> },
  { path: '/fanqie', label: '番茄小说', icon: <BookOpen24Regular /> },
  { path: '/distributed', label: '分布式', icon: <Server24Regular /> },
  { path: '/pipelines', label: '管道', icon: <Pipeline24Regular /> },
]

type SystemStatusType = 'running' | 'stopped' | 'error' | 'unknown'

interface AppLayoutProps {
  extensions?: ExtensionWebUIConfig[]
  systemStatus?: SystemStatusType
}

export function AppLayout({ extensions = [], systemStatus = 'unknown' }: AppLayoutProps) {
  const styles = useStyles()
  const location = useLocation()
  const { themeMode, toggleTheme } = useThemeStore()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  const closeDrawer = () => setDrawerOpen(false)

  const isActive = (path: string) => {
    if (path === '/') return location.pathname === '/'
    return location.pathname.startsWith(path)
  }

  const statusClass = (() => {
    switch (systemStatus) {
      case 'running': return styles.statusRunning
      case 'stopped': return styles.statusWarning
      case 'error': return styles.statusError
      default: return styles.statusUnknown
    }
  })()

  const statusLabel = (() => {
    switch (systemStatus) {
      case 'running': return '运行中'
      case 'stopped': return '已停止'
      case 'error': return '错误'
      default: return '未知'
    }
  })()

  const renderNavItems = (itemClassName: string, activeClassName: string, onClick?: () => void) => (
    <>
      {builtInNavItems.map((item) => (
        <Link
          key={item.path}
          to={item.path}
          className={mergeClasses(
            itemClassName,
            isActive(item.path) && activeClassName,
            sidebarCollapsed && styles.navItemCollapsed,
          )}
          onClick={onClick}
        >
          <span className={styles.navItemIcon}>{item.icon}</span>
          {!sidebarCollapsed && <span className={styles.navItemLabel}>{item.label}</span>}
        </Link>
      ))}
    </>
  )

  const renderExtensionItems = (itemClassName: string, activeClassName: string, onClick?: () => void) => {
    if (extensions.length === 0) return null

    return (
      <div className={styles.sidebarSection}>
        <div className={styles.sidebarSectionLabel}>扩展</div>
        {extensions.map((ext) => {
          const extPath = `/extension/${ext.extension_name}`
          return (
            <Link
              key={ext.extension_name}
              to={extPath}
              className={mergeClasses(
                itemClassName,
                isActive(extPath) && activeClassName,
                sidebarCollapsed && styles.navItemCollapsed,
              )}
              onClick={onClick}
            >
              <span className={styles.extensionIcon}>
                <PuzzlePiece24Regular />
              </span>
              {!sidebarCollapsed && <span className={styles.navItemLabel}>{ext.title}</span>}
            </Link>
          )
        })}
      </div>
    )
  }

  return (
    <div className={styles.root}>
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <Button
            className={styles.menuButton}
            appearance="subtle"
            icon={<Navigation24Regular />}
            onClick={() => setDrawerOpen(true)}
            aria-label="打开导航菜单"
          />
          <Text className={styles.headerLogo}>PPC10</Text>
        </div>

        <div className={styles.headerRight}>
          <Tooltip content={sidebarCollapsed ? '展开侧边栏' : '折叠侧边栏'} relationship="label">
            <Button
              appearance="subtle"
              icon={sidebarCollapsed ? <PanelLeftExpand24Regular /> : <PanelLeftContract24Regular />}
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              aria-label="切换侧边栏"
              className={styles.sidebarToggle}
            />
          </Tooltip>

          <Tooltip content={`系统状态: ${statusLabel}`} relationship="label">
            <span
              className={mergeClasses(styles.statusDot, statusClass)}
              role="status"
              aria-label={statusLabel}
            />
          </Tooltip>

          <Tooltip content={themeMode === 'light' ? '切换深色主题' : themeMode === 'dark' ? '切换高对比度主题' : '切换浅色主题'} relationship="label">
            <Button
              appearance="subtle"
              icon={themeMode === 'light' ? <WeatherMoon24Regular /> : <WeatherSunny24Regular />}
              onClick={toggleTheme}
              aria-label="切换主题"
            />
          </Tooltip>
        </div>
      </header>

      <div className={styles.body}>
        <nav className={mergeClasses(styles.sidebar, sidebarCollapsed && styles.sidebarCollapsed)} aria-label="主导航">
          {renderNavItems(styles.navItem, styles.navItemActive)}
          {renderExtensionItems(styles.navItem, styles.navItemActive)}
        </nav>

        <main className={styles.main}>
          <Outlet />
        </main>
      </div>

      <Drawer
        open={drawerOpen}
        onOpenChange={(_, { open }) => setDrawerOpen(open)}
        separator
        size="small"
      >
        <DrawerHeader>
          <DrawerHeaderTitle
            action={
              <Button
                appearance="subtle"
                icon={<Dismiss24Regular />}
                onClick={closeDrawer}
                aria-label="关闭导航菜单"
              />
            }
          >
            <Text weight="semibold">PPC10</Text>
          </DrawerHeaderTitle>
        </DrawerHeader>
        <DrawerBody>
          <nav aria-label="主导航">
            {renderNavItems(styles.drawerNavItem, styles.drawerNavItemActive, closeDrawer)}
            {renderExtensionItems(styles.drawerNavItem, styles.drawerNavItemActive, closeDrawer)}
          </nav>
        </DrawerBody>
      </Drawer>
    </div>
  )
}