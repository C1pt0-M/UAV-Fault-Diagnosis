<template>
  <div class="app-shell" :class="{ 'is-collapsed': collapsed }">
    <!-- 左侧导航栏 -->
    <aside class="shell-sidebar">

      <!-- 账户区（顶部，可展开时显示完整信息） -->
      <div class="sidebar-account">
        <div class="sidebar-account__main">
          <!-- Logo 始终显示 -->
          <div class="sidebar-account__logo-wrap">
            <img class="sidebar-account__logo" :src="brandLogo" :alt="APP_NAME" />
          </div>

          <!-- 展开时显示的文字信息 -->
          <div class="sidebar-account__info">
            <div class="sidebar-account__name-row">
              <span class="sidebar-account__name">{{ APP_NAME }}</span>
              <el-icon class="sidebar-account__dropdown-icon"><ArrowDown /></el-icon>
            </div>
            <span class="sidebar-account__meta">{{ roleLabel }} · {{ displayName }}</span>
          </div>
        </div>
      </div>

      <!-- 导航 -->
      <nav class="shell-nav">
        <template v-for="(group, idx) in groupedNavigation" :key="idx">
          <!-- 分组标题 (仅在展开时显示) -->
          <div v-if="group.title && !collapsed" class="nav-group-title">
            {{ group.title }}
          </div>
          <!-- 分割线 (仅在折叠且不是第一个分组时显示) -->
          <div v-else-if="group.title && collapsed && idx > 0" class="nav-group-divider"></div>

          <router-link
            v-for="item in group.items"
            :key="item.key"
            :to="item.path"
            class="nav-item"
            :class="{ 'is-active': isActive(item) }"
            :title="collapsed ? item.label : undefined"
          >
            <el-icon class="nav-item__icon"><component :is="item.icon" /></el-icon>
            <span class="nav-item__label">{{ item.label }}</span>
          </router-link>
        </template>
      </nav>

      <!-- 底部：版本 + 登出 -->
      <div class="sidebar-footer">
        <div class="sidebar-footer__main">
          <button class="sidebar-footer__logout" type="button" :title="collapsed ? '退出登录' : undefined" @click="logout">
            <el-icon><SwitchButton /></el-icon>
            <span class="sidebar-footer__logout-label">退出登录</span>
          </button>
          <span class="sidebar-footer__version">v2.0</span>
        </div>

        <!-- 折叠/展开按钮 -->
        <button
          class="sidebar-toggle"
          type="button"
          :title="collapsed ? '展开侧边栏' : '折叠侧边栏'"
          @click="collapsed = !collapsed"
        >
          <el-icon><ArrowLeft v-if="!collapsed" /><ArrowRight v-else /></el-icon>
        </button>
      </div>
    </aside>

    <!-- 主区域 -->
    <div class="shell-main">
      <!-- 顶部标题栏 -->
      <header class="shell-header">
        <div class="shell-header__left">
          <span class="shell-breadcrumb">{{ currentMeta.title }}</span>
          <p v-if="currentMeta.description" class="shell-header__desc">{{ currentMeta.description }}</p>
        </div>

        <div class="shell-header__right">
          <button class="theme-toggle" type="button" :title="isDarkMode ? '切换到浅色模式' : '切换到深色模式'" @click="toggleTheme">
            <el-icon><component :is="isDarkMode ? 'Sunny' : 'Moon'" /></el-icon>
          </button>
          <div class="sys-status">
            <span class="status-dot status-dot--ok"></span>
            <span class="sys-status__label">系统在线</span>
          </div>
        </div>
      </header>

      <!-- 内容区 -->
      <main class="shell-content">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import ElIcon from 'element-plus/es/components/icon/index.mjs'
import { ArrowLeft, ArrowRight, SwitchButton, Sunny, Moon, ArrowDown } from '@element-plus/icons-vue'
import { useRoute, useRouter } from 'vue-router'
import { authApi } from '@/shared/services'
import { APP_NAME } from '@/shared/constants/brand'
import { navigationItems, resolvePageMeta } from '@/shared/constants/navigation'
import {
  clearStoredSession,
  getDisplayName,
  getRoleLabel,
  parseStoredUser
} from '@/shared/utils/user'

const brandLogo = new URL('../../uav_fault_diagnosis_logo.png', import.meta.url).href

const route = useRoute()
const router = useRouter()

// 从 localStorage 恢复折叠状态

// 主题状态
const isDarkMode = ref(localStorage.getItem('theme-mode') === 'dark')

function applyTheme(isDark) {
  if (isDark) {
    document.documentElement.setAttribute('data-theme', 'dark')
  } else {
    document.documentElement.removeAttribute('data-theme')
  }
}
applyTheme(isDarkMode.value)

function toggleTheme() {
  isDarkMode.value = !isDarkMode.value
  localStorage.setItem('theme-mode', isDarkMode.value ? 'dark' : 'light')
  applyTheme(isDarkMode.value)
}

const collapsed = ref(localStorage.getItem('sidebar-collapsed') === 'true')

// 同步折叠状态到 localStorage
watch(collapsed, (val) => {
  localStorage.setItem('sidebar-collapsed', String(val))
})

const currentMeta = computed(() => resolvePageMeta(route.path))
const storedUser = computed(() => parseStoredUser(localStorage.getItem('user')))
const displayName = computed(() => getDisplayName(storedUser.value))
const roleLabel = computed(() => getRoleLabel(storedUser.value))


const groupedNavigation = computed(() => {
  const coreKeys = ['overview', 'monitoring', 'diagnosis']
  const dataKeys = ['assets', 'reports']
  const aiKeys = ['ai']
  const systemKeys = ['settings']

  return [
    {
      title: '核心业务',
      items: navigationItems.filter(i => coreKeys.includes(i.key))
    },
    {
      title: '数据与资产',
      items: navigationItems.filter(i => dataKeys.includes(i.key))
    },
    {
      title: '智能助手',
      items: navigationItems.filter(i => aiKeys.includes(i.key))
    },
    {
      title: '系统与工具',
      items: navigationItems.filter(i => systemKeys.includes(i.key))
    }
  ]
})

function isActive(item) {
  return item.matchPaths.some((path) => route.path === path || route.path.startsWith(`${path}/`))
}

async function logout() {
  try {
    await authApi.logout()
  } catch {
    // local state cleanup is sufficient
  } finally {
    clearStoredSession()
    router.push('/login')
  }
}
</script>

<style scoped>
/* ── 整体布局 ─────────────────────────────────────────────────── */
.app-shell {
  display: grid;
  grid-template-columns: 232px minmax(0, 1fr);
  min-height: 100vh;
  background: var(--bg-canvas);
  transition: grid-template-columns 0.22s cubic-bezier(0.4, 0, 0.2, 1);
}

.app-shell.is-collapsed {
  grid-template-columns: 56px minmax(0, 1fr);
}

/* ── 左侧导航栏 ──────────────────────────────────────────────── */
.shell-sidebar {
  display: flex;
  flex-direction: column;
  background: var(--bg-panel);
  border-right: 1px solid var(--line-soft);
  position: sticky;
  top: 0;
  height: 100vh;
  overflow: hidden;
  flex-shrink: 0;
  width: 232px;
  transition: width 0.22s cubic-bezier(0.4, 0, 0.2, 1);
}

.is-collapsed .shell-sidebar {
  width: 56px;
}

/* ── 账户区 ──────────────────────────────────────────────────── */
.sidebar-account {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 12px;
  border-bottom: 1px solid var(--line-soft);
  flex-shrink: 0;
  min-height: 74px;
}

.sidebar-account__main {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
  flex: 1;
}

.sidebar-account__logo-wrap {
  width: 36px;
  height: 36px;
  border-radius: 8px;
  background: #111827;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  overflow: hidden;
}

.sidebar-account__logo {
  width: 24px;
  height: 24px;
  object-fit: contain;
  /* 白底 logo 在暗背景上 */
  filter: brightness(0) invert(1);
  opacity: 0.9;
}

.sidebar-account__info {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
  overflow: hidden;
  /* 折叠时隐藏 */
  opacity: 1;
  width: auto;
  transition: opacity 0.15s, width 0.22s;
}

.is-collapsed .sidebar-account__info {
  opacity: 0;
  width: 0;
  pointer-events: none;
}

.sidebar-account__name {
  font-size: 12px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: 0;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  line-height: 1.25;
}

.sidebar-account__meta {
  font-size: 11px;
  color: var(--text-tertiary);
  overflow: hidden;
  text-overflow: ellipsis;
}

/* 折叠/展开按钮 */
.sidebar-toggle {
  width: 24px;
  height: 24px;
  border-radius: var(--radius-md);
  border: 1px solid var(--line-soft);
  background: var(--bg-panel-raised);
  color: var(--text-tertiary);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  flex-shrink: 0;
  font-size: 12px;
  transition: border-color 0.15s, color 0.15s, background 0.15s;
}

.sidebar-toggle:hover {
  border-color: var(--brand);
  color: var(--brand);
  background: var(--brand-glow);
}

/* ── 导航 ────────────────────────────────────────────────────── */
.shell-nav {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 18px 14px;
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
}

.is-collapsed .shell-nav {
  align-items: center;
  padding: 18px 0;
}

.nav-group-title {
  margin: 14px 0 6px 8px;
  font-size: 11px;
  font-weight: 600;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  white-space: nowrap;
}

.nav-group-divider {
  height: 1px;
  background: var(--line-faint);
  margin: 8px 4px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 11px 14px;
  border-radius: var(--radius-lg);
  color: var(--text-secondary);
  font-size: 14px;
  font-weight: 500;
  text-decoration: none;
  transition: all 0.2s ease-in-out;
  white-space: nowrap;
  overflow: hidden;
  min-width: 0;
}

.nav-item:hover {
  color: var(--text-primary);
  background: var(--bg-accent-soft);
  transform: translateX(2px);
}

.nav-item.is-active {
  color: var(--brand-strong);
  background: var(--brand-soft);
  font-weight: 600;
}

.is-collapsed .nav-item:hover {
  transform: translateY(-1px);
}

.nav-item__icon {
  font-size: 16px;
  flex-shrink: 0;
}

.nav-item__label {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  opacity: 1;
  transition: opacity 0.15s;
}

.is-collapsed .nav-item__label {
  opacity: 0;
  width: 0;
  flex: none;
  pointer-events: none;
}

/* 折叠时 nav-item 居中显示图标 */
.is-collapsed .nav-item {
  width: 36px;
  height: 36px;
  justify-content: center;
  gap: 0;
  padding: 0;
  border-left: none;
}

.is-collapsed .nav-item.is-active {
  border-left: none;
  box-shadow: inset 2px 0 0 var(--brand);
}

/* ── 底部 ─────────────────────────────────────────────────────── */
.sidebar-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 12px;
  border-top: 1px solid var(--line-soft);
  flex-shrink: 0;
}

.sidebar-footer__logout {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 8px;
  border-radius: var(--radius-md);
  border: none;
  background: transparent;
  color: var(--text-tertiary);
  font-size: 12px;
  cursor: pointer;
  transition: color 0.15s, background 0.15s;
  white-space: nowrap;
  overflow: hidden;
  min-width: 0;
}

.sidebar-footer__logout:hover {
  color: var(--status-danger);
  background: var(--status-danger-soft);
}

.sidebar-footer__logout-label {
  opacity: 1;
  transition: opacity 0.15s;
}

.is-collapsed .sidebar-footer__logout-label {
  opacity: 0;
  width: 0;
  pointer-events: none;
}

.is-collapsed .sidebar-footer__logout {
  justify-content: center;
  padding: 5px;
}

.sidebar-footer__version {
  font-size: 10px;
  color: var(--text-disabled);
  white-space: nowrap;
  flex-shrink: 0;
  opacity: 1;
  transition: opacity 0.15s;
}

.is-collapsed .sidebar-footer__version {
  opacity: 0;
  width: 0;
  overflow: hidden;
}

/* ── 主区域 ─────────────────────────────────────────────────── */
.shell-main {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  overflow: hidden;
}

.shell-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 0 32px;
  height: 56px;
  background: var(--bg-panel);
  border-bottom: 1px solid var(--line-soft);
  flex-shrink: 0;
}

.shell-header__left {
  display: flex;
  align-items: baseline;
  gap: 12px;
  min-width: 0;
}

.shell-breadcrumb {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  white-space: nowrap;
}

.shell-header__desc {
  margin: 0;
  font-size: 12px;
  color: var(--text-tertiary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 360px;
}

.shell-header__right {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
}

.sys-status {
  display: flex;
  align-items: center;
  gap: 5px;
}

.sys-status__label {
  font-size: 12px;
  color: var(--status-ok);
  font-weight: 500;
}

.shell-content {
  flex: 1;
  padding: 28px 32px;
  overflow-y: auto;
  background: var(--bg-canvas);
}

.theme-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: var(--radius-md);
  border: 1px solid var(--line-soft);
  background: var(--bg-panel);
  color: var(--text-secondary);
  cursor: pointer;
  transition: all 0.2s;
  margin-right: var(--space-md);
}
.theme-toggle:hover {
  background: var(--bg-accent);
  color: var(--brand);
}

.sidebar-account__name-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 4px;
}
.sidebar-account__dropdown-icon {
  font-size: 12px;
  color: var(--text-tertiary);
  cursor: pointer;
}
.sidebar-account__main:hover .sidebar-account__dropdown-icon {
  color: var(--brand);
}


.sidebar-footer__main {
  display: flex;
  align-items: center;
  gap: 8px;
  overflow: hidden;
}
.is-collapsed .sidebar-footer__main {
  opacity: 0;
  width: 0;
  pointer-events: none;
  gap: 0;
}
</style>
