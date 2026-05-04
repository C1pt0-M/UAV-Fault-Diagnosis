# 企业级主题系统指南

## 概述

本项目采用简洁专业的企业级主题系统，支持浅色和深色两种模式，为无人机故障检测系统提供一致、专业的界面设计。

## 主题文件

### 核心样式文件

#### `src/styles/enterprise-theme.css`
企业级极简主题样式，包含：
- **完整的CSS变量系统**：颜色、字体、间距、阴影等设计token
- **浅色/深色模式支持**：通过 `data-theme` 属性切换
- **组件样式库**：卡片、按钮、表格、标签、统计卡片等
- **响应式布局**：移动端适配
- **工具类**：快速布局和样式类

#### `src/styles/responsive.css`
响应式布局优化，包含：
- **网格系统**：12列响应式网格
- **断点系统**：xs、sm、md、lg、xl、xxl
- **显示/隐藏**：响应式显示控制
- **间距系统**：响应式padding和margin
- **组件适配**：卡片、表格、按钮、输入框的移动端优化
- **触摸优化**：触摸设备的交互优化
- **打印样式**：打印时的样式优化

#### `src/styles/dark-theme.css`
深色主题样式文件，提供深色模式的特定样式。

### 工具文件

#### `src/utils/theme.js`
简化的主题管理工具，提供：
- `setTheme(themeName)` - 设置主题（light/dark）
- `getCurrentTheme()` - 获取当前主题
- `initTheme()` - 初始化主题（从本地存储或系统偏好加载）
- `toggleTheme()` - 切换主题

## 主题变量系统

### 颜色体系

#### 中性色
```css
--color-white: #ffffff
--color-gray-50: #fafafa
--color-gray-100: #f5f5f5
--color-gray-200: #e5e5e5
--color-gray-300: #d4d4d4
--color-gray-400: #a3a3a3
--color-gray-500: #737373
--color-gray-600: #525252
--color-gray-700: #404040
--color-gray-800: #262626
--color-gray-900: #171717
```

#### 强调色
```css
--color-primary: #1890ff      /* 主色调 */
--color-primary-hover: #40a9ff
--color-success: #52c41a      /* 成功 */
--color-warning: #faad14      /* 警告 */
--color-error: #ff4d4f        /* 错误 */
--color-info: #1890ff         /* 信息 */
```

#### 文字颜色
```css
--text-primary: #171717       /* 主要文字 */
--text-secondary: #525252     /* 次要文字 */
--text-tertiary: #a3a3a3      /* 辅助文字 */
--text-quaternary: #d4d4d4    /* 占位文字 */
```

#### 背景颜色
```css
--bg-primary: #ffffff         /* 主背景 */
--bg-secondary: #fafafa       /* 次要背景 */
--bg-tertiary: #f5f5f5        /* 第三背景 */
--bg-elevated: #ffffff        /* 悬浮背景 */
```

#### 边框颜色
```css
--border-color: #e5e5e5
--border-color-light: #f0f0f0
--border-color-dark: #d9d9d9
```

### 阴影系统
```css
--shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05)
--shadow-base: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1)
--shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)
--shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1)
```

### 圆角系统
```css
--radius-sm: 4px
--radius-md: 6px
--radius-lg: 8px
--radius-xl: 12px
--radius-2xl: 16px
```

### 间距系统
```css
--spacing-xs: 4px
--spacing-sm: 8px
--spacing-md: 12px
--spacing-lg: 16px
--spacing-xl: 20px
--spacing-2xl: 24px
--spacing-3xl: 32px
```

### 字体系统
```css
--font-size-xs: 12px
--font-size-sm: 13px
--font-size-base: 14px
--font-size-lg: 16px
--font-size-xl: 18px
--font-size-2xl: 20px
--font-size-3xl: 24px
--font-size-4xl: 30px
```

### 过渡动画
```css
--transition-fast: 150ms ease
--transition-base: 200ms ease
--transition-slow: 300ms ease
```

## 深色模式

深色模式通过在 `html` 元素上设置 `data-theme="dark"` 属性来激活。

### 深色模式变量
深色模式下，以下变量会自动更新：
- 文字颜色变为浅色以确保对比度
- 背景颜色变为深色
- 边框颜色调整为深色环境
- 阴影效果增强

### 切换主题

#### 方法一：使用主题管理工具
```javascript
import { initTheme, toggleTheme, setTheme, THEME } from '@/utils/theme'

// 初始化主题（自动加载保存的主题或系统偏好）
initTheme()

// 手动切换主题
toggleTheme()

// 设置特定主题
setTheme(THEME.LIGHT)
setTheme(THEME.DARK)
```

#### 方法二：在Vue组件中使用
```vue
<template>
  <div>
    <el-switch
      v-model="isDark"
      @change="handleThemeChange"
      active-text="深色"
      inactive-text="浅色"
    />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { setTheme, THEME, initTheme } from '@/utils/theme'

const isDark = ref(false)

// 初始化主题
initTheme()

// 检查当前主题
isDark.value = document.documentElement.getAttribute('data-theme') === 'dark'

const handleThemeChange = (value) => {
  setTheme(value ? THEME.DARK : THEME.LIGHT)
}
</script>
```

## 组件样式

### 卡片
```html
<div class="card">
  <div class="card-header">
    <div class="card-title">卡片标题</div>
  </div>
  <div class="card-body">
    卡片内容
  </div>
</div>
```

### 按钮
```html
<!-- 基础按钮 -->
<button class="btn">按钮</button>

<!-- 主要按钮 -->
<button class="btn btn-primary">主要按钮</button>

<!-- 按钮尺寸 -->
<button class="btn btn-sm">小按钮</button>
<button class="btn btn-lg">大按钮</button>
```

### 表格
```html
<table class="table">
  <thead>
    <tr>
      <th>列1</th>
      <th>列2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>数据1</td>
      <td>数据2</td>
    </tr>
  </tbody>
</table>
```

### 标签
```html
<span class="tag tag-success">成功</span>
<span class="tag tag-warning">警告</span>
<span class="tag tag-error">错误</span>
<span class="tag tag-info">信息</span>
```

### 统计卡片
```html
<div class="stat-card">
  <div class="stat-value">1,234</div>
  <div class="stat-label">故障检测数</div>
  <div class="stat-change up">
    <span>↑ 12%</span>
  </div>
</div>
```

## 工具类

### 文字对齐
```html
<div class="text-center">居中</div>
<div class="text-left">左对齐</div>
<div class="text-right">右对齐</div>
```

### 文字颜色
```html
<div class="text-primary">主要文字</div>
<div class="text-secondary">次要文字</div>
<div class="text-tertiary">辅助文字</div>
```

### 字体粗细
```html
<div class="font-bold">粗体</div>
<div class="font-medium">中等粗细</div>
```

### 间距（Margin）
```html
<div class="mt-1">上边距 small</div>
<div class="mt-2">上边距 medium</div>
<div class="mt-3">上边距 large</div>
<div class="mt-4">上边距 xlarge</div>

<div class="mb-1">下边距 small</div>
<div class="ml-1">左边距 small</div>
<div class="mr-1">右边距 small</div>
```

### 间距（Padding）
```html
<div class="p-1">内边距 small</div>
<div class="p-2">内边距 medium</div>
<div class="p-3">内边距 large</div>
<div class="p-4">内边距 xlarge</div>
```

### Flex布局
```html
<div class="flex">Flex容器</div>
<div class="flex flex-col">垂直Flex</div>
<div class="flex items-center">垂直居中</div>
<div class="flex justify-between">两端对齐</div>
<div class="flex justify-center">水平居中</div>
<div class="flex gap-2">间距</div>
```

### 尺寸
```html
<div class="w-full">全宽</div>
<div class="h-full">全高</div>
```

### 圆角
```html
<div class="rounded">圆角</div>
<div class="rounded-lg">大圆角</div>
<div class="rounded-xl">超大圆角</div>
```

### 阴影
```html
<div class="shadow">基础阴影</div>
<div class="shadow-lg">大阴影</div>
```

## 响应式设计

### 断点系统
- `xs`: < 576px
- `sm`: ≥ 576px
- `md`: ≥ 768px
- `lg`: ≥ 992px
- `xl`: ≥ 1200px
- `xxl`: ≥ 1600px

### 响应式网格
```html
<div class="row">
  <div class="col-12 col-md-6 col-lg-4">响应式列</div>
  <div class="col-12 col-md-6 col-lg-4">响应式列</div>
  <div class="col-12 col-md-6 col-lg-4">响应式列</div>
</div>
```

### 响应式显示
```html
<div class="d-none d-md-block">仅在中屏及以上显示</div>
<div class="d-md-none">仅在小屏显示</div>
```

## 浏览器兼容性

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 性能建议

1. **按需引入**：只引入需要的样式文件
2. **CSS变量**：充分利用CSS变量实现主题切换，避免重复代码
3. **减少重绘**：避免频繁修改DOM样式
4. **图片优化**：使用WebP格式图片
5. **懒加载**：对非首屏内容使用懒加载

## 无障碍支持

- 支持高对比度模式
- 支持减少动画模式
- 语义化HTML标签
- 键盘导航支持
- 适当的颜色对比度（WCAG 2.1 AA标准）

## 设计原则

1. **简洁专业**：采用企业级设计风格，避免过度装饰
2. **一致性**：保持整个应用的视觉一致性
3. **可读性**：确保文字和背景的对比度符合可访问性标准
4. **响应式**：适配不同屏幕尺寸和设备
5. **性能优先**：优化渲染性能，提供流畅的用户体验

## 技术支持

如有问题或建议，请联系开发团队。
