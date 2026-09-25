# WinUI Web Library

WinUI-on-Web 的独立可复用组件库（Vue 3 + TypeScript）。将微软 WinUI 控件以 **XAML 风格属性** 的方式带到 Web，含明暗主题令牌、动效与内置图标字体。

> 独立库项目，由 `WinUIonWeb`（同仓库示例应用）整理重构而来，现有 gallery 保持不动。

## 特性
- 控件 props 使用 XAML 风格 **PascalCase** 命名（`Content`、`IsEnabled`、`Style`、`CornerRadius`…）。
- 内置 light/dark 设计令牌系统与动画时长变量。
- 内置图标字体 `WinUIOnWebIcons`。
- 库模式构建，输出 ESM + CJS + 类型声明。

## 安装
本地引用：
```jsonc
// 使用方 package.json
{
  "dependencies": {
    "vue": "^3.5",
    "winui-web-library": "file:../WinUI-Web-Library"
  }
}
```
（发布至 npm 后可用包名安装。）

## 快速开始
```ts
// 入口：import 库一次即可带出样式令牌 + 字体
import { createApp } from 'vue'
import App from './App.vue'
import 'winui-web-library'
// 设置明暗主题（需写主题 class，否则控件落到无令牌态）
document.documentElement.classList.add('theme-light') // 或 theme-dark
```

若使用依赖 i18n 的控件（如 `WinTextBlock`），启动时 provide i18n：
```ts
import { createI18n, i18nKey, normalizeLocale } from 'winui-web-library'
const app = createApp(App)
app.provide(i18nKey, createI18n(normalizeLocale(navigator.language)))
app.mount('#app')
```

在组件中使用：
```vue
<script setup lang="ts">
import {
  WinButton, WinTextBlock, WinCheckBox,
  WinToggleSwitch, WinStackPanel,
} from 'winui-web-library'
import { ref } from 'vue'
const count = ref(0)
const checked = ref(true)
</script>

<template>
  <WinStackPanel Spacing="16">
    <WinTextBlock FontSize="28" FontWeight="600" Text="Hello WinUI" />
    <WinButton Style="AccentButtonStyle" :Content="`Clicked ${count}`" @Click="count++" />
    <WinCheckBox v-model="checked" Content="Check me" />
    <WinToggleSwitch v-model="checked" Header="Toggle" />
  </WinStackPanel>
</template>
```

## 主题
在根元素（`<html>`）切换 class 即可全局切换明暗令牌：
```js
const el = document.documentElement
el.classList.toggle('theme-dark', isDark)
el.classList.toggle('theme-light', !isDark)
```

## 控件清单（当前已迁移）
| 控件 | 说明 |
|---|---|
| `WinButton` | Default / Accent / Subtle 三种内建样式按钮 |
| `WinTextBlock` | 文本块 |
| `WinCheckBox` | 复选框（支持三态） |
| `WinToggleSwitch` | 开关 |
| `WinToggleButton` | 切换按钮 |
| `WinStackPanel` | 纵向/横向堆叠面板 |
| `WinGrid` | 网格布局（支持 ColumnDefinitions/RowDefinitions） |
| `WinScrollViewer` | 滚动容器 |

> 更多控件按批次从 `WinUIonWeb` 持续迁移中。

## 属性与事件约定
- 属性：PascalCase（`Content`/`IsEnabled`/`Style`/`Background`/`BorderThickness`/`Padding`/`Margin`/`Width`/`Height`/`FontSize`/`FontWeight`…）。
- `Padding`/`Margin`/`BorderThickness` 支持 XAML Thickness：`"1"`、`"1,2"`、`"1,2,3,4"`。
- 事件：`@Click`、`@update:IsChecked`、`v-model:Text` 等。

## 开发
```sh
npm install
node_modules/.bin/vite            # dev（dev/ 目录为最小验收样例）
npm run build                      # type-check + build
npm run lint                       # oxlint + eslint
```

注：若全局 npm `script-shell` 被误设为 Node 可执行文件，`npm run <script>` 会报 MODULE_NOT_FOUND（非本项目问题）；此时改走 `node_modules/.bin/vite` / `vue-tsc` 等直跑。

## 目录结构
```
src/
  index.ts              # 库入口：样式 + 具名导出
  components/           # 控件与聚合导出
    i18n/ , Strings/    # 库内 i18n
  styles/
    theme.css           # light/dark 设计令牌
    animations.css      # 动效时长变量
    font.css            # WinUIOnWebIcons @font-face
  assets/fonts/         # SEGOEICONS.TTF
dev/                    # 最小验收样例
```

## 声明
本项目为对 Microsoft WinUI 的 Web 复刻，未使用其任何源代码，为独立实现，与 Microsoft 无关联。