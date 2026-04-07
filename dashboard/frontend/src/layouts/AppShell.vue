<template>
  <div class="app-shell">
    <aside class="shell-sidebar">
      <div class="brand-card">
        <div class="brand-mark">U</div>
        <div>
          <div class="brand-title">智诊天眸</div>
          <div class="brand-subtitle">基于数据驱动的无人机故障检测系统</div>
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
          <button class="shell-user__logout" type="button" @click="logout">
            退出登录
          </button>
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
import { useRoute, useRouter } from 'vue-router'
import { authApi } from '@/shared/services'
import { navigationItems, resolvePageMeta } from '@/shared/constants/navigation'
import {
  clearStoredSession,
  getDisplayName,
  getRoleLabel,
  parseStoredUser
} from '@/shared/utils/user'

const route = useRoute()
const router = useRouter()

const currentMeta = computed(() => resolvePageMeta(route.path))
const storedUser = computed(() => parseStoredUser(localStorage.getItem('user')))
const displayName = computed(() => getDisplayName(storedUser.value))
const roleLabel = computed(() => getRoleLabel(storedUser.value))

function isActive(item) {
  return item.matchPaths.some((path) => route.path === path || route.path.startsWith(`${path}/`))
}

async function logout() {
  try {
    await authApi.logout()
  } catch {
    // local state cleanup is sufficient for frontend session reset
  } finally {
    clearStoredSession()
    router.push('/login')
  }
}
</script>

<style scoped>
.app-shell {
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr);
  gap: 24px;
  min-height: 100vh;
  padding: 24px;
}

.shell-sidebar,
.shell-header,
.shell-content {
  border: 1px solid rgba(0,0,0,0.04);
  border-radius: var(--radius-xl);
  background: var(--bg-panel);
  box-shadow: 0 4px 20px rgba(0,0,0,0.02);
}

.shell-sidebar {
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 24px 20px;
}

.brand-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  border-radius: var(--radius-lg);
  background: linear-gradient(135deg, #11323f 0%, #0f766e 100%);
  color: #f4fbfb;
  box-shadow: 0 8px 24px rgba(15, 118, 110, 0.12);
}

.brand-mark {
  display: grid;
  place-items: center;
  width: 48px;
  height: 48px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.15);
  font-size: 24px;
  font-weight: 800;
}

.brand-title {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.5px;
}

.brand-subtitle {
  margin-top: 6px;
  font-size: 12px;
  line-height: 1.5;
  color: rgba(255, 255, 255, 0.7);
}

.shell-nav {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 16px 18px;
  border-radius: 12px;
  color: var(--text-secondary);
  font-weight: 500;
  text-decoration: none;
  transition: all 0.2s ease;
}

.nav-item:hover {
  color: var(--text-primary);
  background: rgba(16, 50, 64, 0.04);
}

.nav-item.is-active {
  color: #ffffff;
  background: linear-gradient(135deg, #103240 0%, #0f766e 100%);
  box-shadow: 0 4px 12px rgba(15, 118, 110, 0.2);
}

.shell-main {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.shell-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  padding: 32px 36px;
}

.shell-caption {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.18em;
  color: var(--text-tertiary);
  text-transform: uppercase;
}

.shell-title {
  margin: 12px 0 0;
  font-size: 32px;
  font-weight: 700;
  letter-spacing: -0.5px;
  color: var(--text-primary);
  line-height: 1.2;
}

.shell-description {
  margin: 12px 0 0;
  max-width: 600px;
  font-size: 15px;
  line-height: 1.6;
  color: var(--text-secondary);
}

.shell-user {
  display: flex;
  align-items: center;
  gap: 16px;
}

.shell-user__status {
  padding: 8px 16px;
  border-radius: 999px;
  background: rgba(22, 121, 76, 0.1);
  color: var(--success);
  font-size: 13px;
  font-weight: 600;
}

.shell-user__card {
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 10px 16px;
  border-radius: 12px;
  background: #f8f9fa;
  border: 1px solid rgba(0,0,0,0.04);
}

.shell-user__card strong {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.shell-user__card small {
  margin-top: 2px;
  font-size: 12px;
  color: var(--text-tertiary);
}

.shell-user__logout {
  padding: 12px 20px;
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 12px;
  background: #ffffff;
  color: var(--text-primary);
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.shell-user__logout:hover {
  background: #f1f3f5;
}

.shell-content {
  padding: 36px;
  flex: 1;
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
