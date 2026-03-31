<template>
  <div class="app-shell">
    <aside class="shell-sidebar">
      <div class="brand-card">
        <div class="brand-mark">U</div>
        <div>
          <div class="brand-title">故障诊断系统</div>
          <div class="brand-subtitle">无人机数据驱动监测与诊断工作台</div>
        </div>
      </div>

      <nav class="shell-nav">
        <router-link
          v-for="item in navigationItems"
          :key="item.key"
          :to="item.path"
          class="nav-item"
          :class="{ 'is-active': isActive(item) }"
        >
          <el-icon><component :is="item.icon" /></el-icon>
          <span>{{ item.label }}</span>
        </router-link>
      </nav>
    </aside>

    <div class="shell-main">
      <header class="shell-header">
        <div>
          <div class="shell-caption">UAV FAULT DETECTION WORKSPACE</div>
          <h1 class="shell-title">{{ currentMeta.title }}</h1>
          <p class="shell-description">{{ currentMeta.description }}</p>
        </div>

        <div class="shell-user">
          <span class="shell-user__status">系统在线</span>
          <div class="shell-user__card">
            <strong>{{ displayName }}</strong>
            <small>{{ roleLabel }}</small>
          </div>
        </div>
      </header>

      <main class="shell-content">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import ElIcon from 'element-plus/es/components/icon/index.mjs'
import { useRoute } from 'vue-router'
import { navigationItems, resolvePageMeta } from '@/shared/constants/navigation'
import { getDisplayName, getRoleLabel, parseStoredUser } from '@/shared/utils/user'

const route = useRoute()

const currentMeta = computed(() => resolvePageMeta(route.path))
const storedUser = computed(() => parseStoredUser(localStorage.getItem('user')))
const displayName = computed(() => getDisplayName(storedUser.value))
const roleLabel = computed(() => getRoleLabel(storedUser.value))

function isActive(item) {
  return item.matchPaths.some((path) => route.path === path || route.path.startsWith(`${path}/`))
}
</script>

<style scoped>
.app-shell {
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr);
  gap: 18px;
  min-height: 100vh;
  padding: 18px;
}

.shell-sidebar,
.shell-header,
.shell-content {
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-xl);
  background: var(--bg-panel);
  box-shadow: var(--shadow-soft);
}

.shell-sidebar {
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 20px 18px;
}

.brand-card {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 18px;
  border-radius: var(--radius-lg);
  background: linear-gradient(135deg, #11323f 0%, #0f766e 100%);
  color: #f4fbfb;
}

.brand-mark {
  display: grid;
  place-items: center;
  width: 48px;
  height: 48px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.14);
  font-size: 24px;
  font-weight: 800;
}

.brand-title {
  font-size: 18px;
  font-weight: 700;
}

.brand-subtitle {
  margin-top: 4px;
  font-size: 12px;
  line-height: 1.6;
  color: rgba(244, 251, 251, 0.82);
}

.shell-nav {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
  border-radius: 16px;
  color: var(--text-secondary);
  text-decoration: none;
  transition: 0.2s ease;
}

.nav-item:hover {
  color: var(--text-primary);
  background: var(--bg-accent-soft);
}

.nav-item.is-active {
  color: #effafa;
  background: linear-gradient(135deg, #0b4f59 0%, #0f766e 100%);
}

.shell-main {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.shell-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 22px 24px;
}

.shell-caption {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.14em;
  color: var(--text-tertiary);
}

.shell-title {
  margin: 10px 0 0;
  font-size: 34px;
  line-height: 1.1;
}

.shell-description {
  margin: 10px 0 0;
  max-width: 720px;
  font-size: 14px;
  line-height: 1.8;
  color: var(--text-secondary);
}

.shell-user {
  display: flex;
  align-items: center;
  gap: 12px;
}

.shell-user__status {
  padding: 10px 14px;
  border-radius: 999px;
  background: #e8f5ef;
  color: var(--success);
  font-size: 13px;
  font-weight: 600;
}

.shell-user__card {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  padding: 12px 14px;
  border-radius: 18px;
  background: #f7fafb;
  border: 1px solid var(--line-soft);
}

.shell-user__card strong {
  font-size: 14px;
}

.shell-user__card small {
  margin-top: 4px;
  color: var(--text-tertiary);
}

.shell-content {
  padding: 24px;
}

@media (max-width: 1280px) {
  .app-shell {
    grid-template-columns: 1fr;
  }

  .shell-user {
    width: 100%;
    justify-content: space-between;
  }
}
</style>
