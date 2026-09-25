# PPC WebUI (Vue 3 + WinUI on Web)

PPC_NEXT 的 Web 管理界面，基于 **Vue 3 + TypeScript + Vite + [WinUI-Web-Library](../WinUIonWeb/WinUI-Web-Library)** 构建，
提供系统状态、配置管理、任务转换、扩展管理、分布式集群与番茄 TTS 等模块的可视化操作。
采用 Windows 11 Fluent 风格控件、明暗主题令牌与本地化图标字体，观感与原生 WinUI 一致。

> 本目录即当前 PPC WebUI 的**唯一边**——旧 React 工程（`@fluentui/react-components`）已被迁移并弃用（备份见仓库根 `../..` 之外）。

## 环境要求

- Node.js `^20.19.0` 或 `>=22.12.0`（见 `package.json#engines`）
- npm（随 Node 附带）

## 安装

```bash
cd webui/WebUI-Vue
npm install
```

依赖通过 `package.json` 中的 `file:` 本地引用指向 `webui/WebUI-Vue/WinUI-Web-Library`（WinUI 组件库，自包含副本），
安装时会一并解析该库。改动 WinUI 库后无需重新 `npm install`。

## 常用命令

```bash
npm run dev          # 启动开发服务器（默认 http://localhost:3000）
npm run build        # type-check（vue-tsc）+ vite build，产出 dist/
npm run type-check   # 仅类型检查（vue-tsc --build）
npm run lint         # oxlint + eslint（含 --fix）
npm run preview      # 本地预览构建产物
```

`npm run dev` 下 `/api` 开头的请求会代理到本地 PPC 服务（`http://127.0.0.1:5000`），
配置见 `vite.config.ts` 的 `server.proxy`。

## 构建产物

`npm run build` 输出到 `./dist/`。Vite 按路由分包：每个功能页（Dashboard / Config / TtsConvert /
Extensions / Distributed / Fanqie）拆成独立 JS/CSS chunk，共享依赖打进 `index-*.js`。

## 部署（生产）

WebUI 静态资源与 `/api/*` 接口由 PPC 后端（Flask）统一托管，无需独立静态服务器：

```bash
ppc webui                     # 后端启动，若 dist 缺失会自动执行 npm run build
ppc webui --host 127.0.0.1 --port 5001 --api-token <token>
```

- 后端通过 `src/web/runner.py` 的 `WEBUI_DIST`（即 `webui/WebUI-Vue/dist`）定位静态目录，
  由 `create_app` 的 SPA 回退路由服务页面。
- `/api/*` 路由保持 JSON 契约不变（`/api` 代理目标与 React 时代一致）；未知 `/api` 路径返回 404 JSON，不会回退到 `index.html`。
- 也可用 `FLASK_STATIC_FOLDER` 环境变量显式指定静态目录；不传时回退查找 `webui/WebUI-Vue/dist`。

## 主题与减少动效

- **明暗 Scheme**：主题模式通过 `<html>` 上的 `theme-light` / `theme-dark` class 与 `colorScheme` 驱动 WinUI 库令牌。
- **模式**：`light` / `dark` / `high-contrast` 三态循环（`AppLayout` 顶栏切换），持久化于 `localStorage` 的 `ppc-theme-mode`。
- **减少动效**：WinUI-Web-Library 的动画样式在 `@media (prefers-reduced-motion: reduce)` 下自动退化为无/微动效，
  随系统设置生效，无需额外配置。

## 目录结构

```
WebUI-Vue/
├─ src/
│  ├─ api/           # 后端 API 客户端（client.ts / types.ts）
│  ├─ composables/   # useApi / useApiAction / useSse / useToast
│  ├─ components/    # 共享与分页组件（AppLayout、ConfirmDialog、StatusBadge…）
│  ├─ pages/         # 六大功能页（*.vue）
│  ├─ theme/         # 主题 composable 与令牌（useTheme / tokens）
│  ├─ constants/     # 端口、断点等常量
│  ├─ styles/        # 全局样式 main.css
│  ├─ App.vue / router.ts / main.ts
├─ vite.config.ts    # /api 代理 + @ 别名
└─ package.json
```