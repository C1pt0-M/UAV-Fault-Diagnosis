import { createRouter, createWebHistory } from 'vue-router'
import { ElMessage } from 'element-plus'
import Layout from '@/components/Layout.vue'
import { authApi, setupApi } from '@/shared/services'
import { clearStoredSession, parseStoredUser, storeSession } from '@/shared/utils/user'

const RegisterRedirect = { name: 'RegisterRedirect', template: '<div />' }

async function redirectRegisterRoute(_to, _from, next) {
  try {
    const response = await setupApi.status()
    next(response.data?.needs_setup ? '/setup' : '/login')
  } catch {
    next('/login')
  }
}

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
    component: RegisterRedirect,
    beforeEnter: redirectRegisterRoute,
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

let profileValidationPromise = null

function isPublicAuthPath(path) {
  return path === '/login' || path === '/setup' || path === '/register'
}

async function validateStoredSession() {
  const token = localStorage.getItem('token')
  const user = parseStoredUser(localStorage.getItem('user'))

  if (!token || !user) return false

  if (!profileValidationPromise) {
    profileValidationPromise = authApi.profile()
      .then((response) => {
        storeSession(token, response.data)
        return true
      })
      .catch(() => {
        clearStoredSession()
        return false
      })
      .finally(() => {
        profileValidationPromise = null
      })
  }

  return profileValidationPromise
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

  if (isPublicAuthPath(to.path) && token && user) {
    next('/')
    return
  }

  next()
})

export default router
