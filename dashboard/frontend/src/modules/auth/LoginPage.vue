<template>
  <div class="login-page">
    <section class="login-card">
      <div class="login-card__hero">
        <span class="login-card__tag">登录入口</span>
        <h1>故障诊断系统</h1>
        <p>
          面向运维与演示人员的统一入口。登录后可进入总览、实时监测、诊断中心、
          资产与报告以及 AI 助手模块。
        </p>
      </div>

      <div class="login-card__status" :class="`is-${connectionStatus}`">
        <span>{{ connectionLabel }}</span>
        <el-button link type="primary" @click="refreshConnection" :disabled="checkingConnection">
          重新检测
        </el-button>
      </div>

      <el-form ref="formRef" :model="form" :rules="rules" label-position="top">
        <el-form-item label="用户名" prop="username">
          <el-input v-model="form.username" placeholder="请输入用户名" />
        </el-form-item>
        <el-form-item label="密码" prop="password">
          <el-input
            v-model="form.password"
            type="password"
            show-password
            placeholder="请输入密码"
            @keyup.enter="submit"
          />
        </el-form-item>
        <el-button type="primary" size="large" class="login-button" :loading="submitting" @click="submit">
          登录系统
        </el-button>
      </el-form>

      <div class="login-card__footer">
        <span>默认管理员账号：admin / admin123</span>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import { ElForm, ElFormItem } from 'element-plus/es/components/form/index.mjs'
import ElInput from 'element-plus/es/components/input/index.mjs'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { authApi } from '@/utils/api'
import { checkBackend, getConnectionStatus } from '@/utils/request'

const router = useRouter()
const formRef = ref(null)
const submitting = ref(false)
const checkingConnection = ref(false)
const connectionStatus = ref(getConnectionStatus())

const form = reactive({
  username: 'admin',
  password: 'admin123'
})

const rules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
}

const connectionLabel = computed(() => {
  switch (connectionStatus.value) {
    case 'connected':
      return '后端连接正常'
    case 'checking':
      return '正在检测后端状态'
    case 'disconnected':
      return '后端未连接，部分能力不可用'
    default:
      return '正在初始化连接状态'
  }
})

async function refreshConnection() {
  checkingConnection.value = true
  try {
    const ok = await checkBackend()
    connectionStatus.value = ok ? 'connected' : 'disconnected'
  } finally {
    checkingConnection.value = false
  }
}

async function submit() {
  if (!formRef.value) return

  await formRef.value.validate(async (valid) => {
    if (!valid) return

    submitting.value = true
    try {
      const response = await authApi.login({
        username: form.username,
        password: form.password
      })

      localStorage.setItem('token', response.data.token)
      localStorage.setItem('user', JSON.stringify(response.data.user))
      ElMessage.success('登录成功')
      router.push('/')
    } catch (error) {
      ElMessage.error(error.message || '登录失败，请检查账号和密码')
    } finally {
      submitting.value = false
    }
  })
}

onMounted(refreshConnection)
</script>

<style scoped>
.login-page {
  display: grid;
  place-items: center;
  min-height: 100vh;
  padding: 32px;
}

.login-card {
  width: min(480px, 100%);
  padding: 34px;
  border-radius: 30px;
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(16, 50, 64, 0.12);
  box-shadow: 0 30px 60px rgba(15, 31, 38, 0.14);
}

.login-card__hero h1 {
  margin: 12px 0 0;
  font-size: 34px;
}

.login-card__hero p {
  margin: 12px 0 0;
  color: var(--text-secondary);
  line-height: 1.8;
}

.login-card__tag {
  display: inline-flex;
  padding: 6px 10px;
  border-radius: 999px;
  background: var(--brand-soft);
  color: var(--brand-strong);
  font-size: 12px;
  font-weight: 700;
}

.login-card__status {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin: 24px 0;
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid var(--line-soft);
}

.login-card__status.is-connected {
  background: #edf8f2;
  color: var(--success);
}

.login-card__status.is-disconnected {
  background: #fff3f2;
  color: var(--danger);
}

.login-card__status.is-checking,
.login-card__status.is-unknown {
  background: #f2f7f8;
  color: var(--text-secondary);
}

.login-button {
  width: 100%;
  margin-top: 8px;
}

.login-card__footer {
  margin-top: 18px;
  color: var(--text-tertiary);
  font-size: 13px;
}
</style>
