import { createRouter, createWebHistory } from 'vue-router'
import { ElMessage } from 'element-plus'
import Layout from '@/components/Layout.vue'
import { parseStoredUser } from '@/shared/utils/user'

const routes = [
  {
    path: '/setup',
    name: 'Setup',
    component: () => import('@/views/Setup.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/register',
    name: 'Register',
    component: () => import('@/views/Register.vue'),
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
        component: () => import('@/views/Home.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'monitoring',
        name: 'Monitoring',
        component: () => import('@/views/Telemetry.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'diagnosis',
        name: 'Diagnosis',
        component: () => import('@/views/FaultDetection.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'model-compare',
        name: 'ModelCompare',
        component: () => import('@/views/ModelCompare.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'assets',
        name: 'Assets',
        component: () => import('@/views/Logs.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'reports',
        name: 'Reports',
        component: () => import('@/views/DiagnosisReports.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'ai',
        name: 'AIAssistant',
        component: () => import('@/views/AIAssistant.vue'),
        meta: { requiresAuth: true }
      }
    ]
  },
  { path: '/telemetry', redirect: '/monitoring' },
  { path: '/fault-detection', redirect: '/diagnosis' },
  { path: '/logs', redirect: '/assets' },
  { path: '/diagnosis-reports', redirect: '/reports' },
  { path: '/ai-assistant', redirect: '/ai' },
  { path: '/data-analysis', redirect: '/diagnosis' },
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
    ElMessage.warning('璇峰厛鐧诲綍')
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

