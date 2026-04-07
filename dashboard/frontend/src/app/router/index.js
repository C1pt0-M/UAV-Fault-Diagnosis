import { createRouter, createWebHistory } from 'vue-router'
import { ElMessage } from 'element-plus'
import Layout from '@/components/Layout.vue'
import { parseStoredUser } from '@/shared/utils/user'

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
    component: () => import('@/modules/auth/RegisterPage.vue'),
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
        path: 'model-compare',
        name: 'ModelCompare',
        component: () => import('@/modules/competition/ModelComparePage.vue'),
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
      }
    ]
  },
  { path: '/:pathMatch(.*)*', redirect: '/' }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  const user = parseStoredUser(localStorage.getItem('user'))

  if (to.meta.requiresAuth && (!token || !user)) {
    ElMessage.warning('请先登录')
    next('/login')
    return
  }

  if ((to.path === '/login' || to.path === '/register') && token && user) {
    next('/')
    return
  }

  next()
})

export default router
