import { createRouter, createWebHistory } from 'vue-router'
import { ElMessage } from 'element-plus'
import Layout from '@/components/Layout.vue'
import { authApi } from '@/shared/services'
import { clearStoredSession, parseStoredUser, storeSession } from '@/shared/utils/user'
import { isNavigationAvailable, navigationItemForPath } from '@/shared/constants/navigation'
import { applyRuntimeCapabilities, runtimeCapabilities } from '@/shared/state/runtimeCapabilities'

const routes = [
  {
    path: '/setup',
    name: 'Setup',
    component: () => import('@/modules/setup/SetupPage.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/modules/auth/LoginPage.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/register',
    name: 'Register',
    component: () => import('@/modules/auth/LoginPage.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/',
    component: Layout,
    meta: { requiresAuth: true },
    children: [
      {
        path: '',
        name: 'Overview',
        component: () => import('@/modules/overview/OverviewPage.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'monitoring',
        name: 'Monitoring',
        component: () => import('@/modules/monitoring/MonitoringPage.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'diagnosis',
        name: 'Diagnosis',
        component: () => import('@/modules/diagnosis/DiagnosisPage.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'assets',
        name: 'Assets',
        component: () => import('@/modules/assets/AssetsPage.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'reports',
        name: 'Reports',
        component: () => import('@/modules/reports/ReportsPage.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'ai',
        name: 'AIAssistant',
        component: () => import('@/modules/ai/AIAssistantPage.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'settings',
        name: 'Settings',
        component: () => import('@/modules/settings/SettingsPage.vue'),
        meta: { requiresAuth: true }
      },
    ]
  },
  { path: '/:pathMatch(.*)*', redirect: '/' }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

let pendingProfileValidation = null

function isPublicAuthPath(path) {
  return path === '/login' || path === '/setup' || path === '/register'
}

async function validateStoredSession() {
  const token = localStorage.getItem('token')
  const user = parseStoredUser(localStorage.getItem('user'))

  if (!token || !user) return false

  if (pendingProfileValidation?.token !== token) {
    const validation = { token, promise: null }
    validation.promise = authApi.profile()
      .then((response) => {
        if (localStorage.getItem('token') !== token) return false
        storeSession(token, response.data)
        applyRuntimeCapabilities(response.capabilities)
        return true
      })
      .catch(() => {
        if (localStorage.getItem('token') === token) clearStoredSession()
        return false
      })
      .finally(() => {
        if (pendingProfileValidation === validation) pendingProfileValidation = null
      })
    pendingProfileValidation = validation
  }

  return pendingProfileValidation.promise
}

router.beforeEach(async (to, from, next) => {
  const token = localStorage.getItem('token')
  const user = parseStoredUser(localStorage.getItem('user'))

  if (to.meta.requiresAuth && (!token || !user)) {
    ElMessage.warning('请先登录')
    next('/login')
    return
  }

  if (token && user) {
    const isValid = await validateStoredSession()
    if (!isValid) {
      if (to.meta.requiresAuth) {
        ElMessage.warning('登录已过期，请重新登录')
        next('/login')
        return
      }
      next()
      return
    }
  }

  const navigationItem = navigationItemForPath(to.path)
  if (navigationItem && !isNavigationAvailable(navigationItem, runtimeCapabilities)) {
    ElMessage.info(`${navigationItem.label}未启用或能力信息不可用，请联系管理员`)
    next('/')
    return
  }

  if (isPublicAuthPath(to.path) && token && user) {
    next('/')
    return
  }

  next()
})

export default router
