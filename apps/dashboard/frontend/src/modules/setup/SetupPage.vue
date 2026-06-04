<template>
  <div class="setup-page">
    <section class="setup-card">
      <div class="setup-card__header">
        <span class="setup-card__tag">System Initialization</span>
        <h1>系统初始化</h1>
        <p>{{ pageDescription }}</p>
      </div>

      <div v-if="loading" class="setup-status">正在检查系统状态...</div>

      <div v-else-if="installed" class="setup-status setup-status--ready">
        <strong>系统已完成初始化</strong>
        <span>请使用已有管理员账号登录。</span>
        <el-button type="primary" size="large" @click="router.push('/login')">进入登录页</el-button>
      </div>

      <el-form v-else ref="formRef" :model="form" :rules="rules" label-position="top" class="setup-form">
        <el-form-item label="管理员账号" prop="username">
          <el-input v-model="form.username" placeholder="请输入管理员账号" />
        </el-form-item>

        <el-form-item label="管理员姓名" prop="real_name">
          <el-input v-model="form.real_name" placeholder="请输入管理员姓名" />
        </el-form-item>

        <el-form-item label="邮箱" prop="email">
          <el-input v-model="form.email" placeholder="请输入邮箱地址" />
        </el-form-item>

        <el-form-item label="密码" prop="password">
          <el-input
            v-model="form.password"
            type="password"
            show-password
            placeholder="请输入 6-20 位密码"
            @keyup.enter="submit"
          />
        </el-form-item>

        <el-form-item label="确认密码" prop="confirmPassword">
          <el-input
            v-model="form.confirmPassword"
            type="password"
            show-password
            placeholder="请再次输入密码"
            @keyup.enter="submit"
          />
        </el-form-item>

        <div class="setup-actions">
          <el-button @click="router.push('/login')">返回登录</el-button>
          <el-button type="primary" :loading="submitting" @click="submit">完成初始化</el-button>
        </div>
      </el-form>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import ElButton from 'element-plus/es/components/button/index.mjs'
import { ElForm, ElFormItem } from 'element-plus/es/components/form/index.mjs'
import ElInput from 'element-plus/es/components/input/index.mjs'
import { ElMessage } from 'element-plus'
import { setupApi } from '@/shared/services'

const router = useRouter()
const formRef = ref(null)
const loading = ref(true)
const installed = ref(false)
const submitting = ref(false)

const form = reactive({
  username: '',
  real_name: '',
  email: '',
  password: '',
  confirmPassword: ''
})

function validateConfirmPassword(_rule, value, callback) {
  if (!value) {
    callback(new Error('请再次输入密码'))
    return
  }
  if (value !== form.password) {
    callback(new Error('两次输入的密码不一致'))
    return
  }
  callback()
}

const rules = {
  username: [
    { required: true, message: '请输入管理员账号', trigger: 'blur' },
    { min: 3, max: 20, message: '账号长度需为 3-20 位', trigger: 'blur' }
  ],
  real_name: [{ required: true, message: '请输入管理员姓名', trigger: 'blur' }],
  email: [
    { required: true, message: '请输入邮箱地址', trigger: 'blur' },
    { type: 'email', message: '请输入有效的邮箱地址', trigger: ['blur', 'change'] }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, max: 20, message: '密码长度需为 6-20 位', trigger: 'blur' }
  ],
  confirmPassword: [{ validator: validateConfirmPassword, trigger: 'blur' }]
}

const pageDescription = computed(() => (
  installed.value
    ? '当前系统已经创建管理员账号，后续访问需要登录验证。'
    : '首次使用前请创建管理员账号，初始化完成后返回登录页进入系统。'
))

async function loadSetupStatus() {
  loading.value = true
  try {
    const response = await setupApi.status()
    installed.value = Boolean(response.data?.is_installed)
  } catch (error) {
    ElMessage.error(error.message || '系统状态检查失败')
  } finally {
    loading.value = false
  }
}

async function submit() {
  if (!formRef.value) return

  await formRef.value.validate(async (valid) => {
    if (!valid) return

    submitting.value = true
    try {
      await setupApi.initialize({
        username: form.username,
        real_name: form.real_name,
        email: form.email,
        password: form.password
      })
      ElMessage.success('系统初始化成功，请登录管理员账号')
      router.push('/login')
    } catch (error) {
      ElMessage.error(error.message || '系统初始化失败')
      if (error.status === 400) {
        await loadSetupStatus()
      }
    } finally {
      submitting.value = false
    }
  })
}

onMounted(loadSetupStatus)
</script>

<style scoped>
.setup-page {
  display: grid;
  place-items: center;
  min-height: 100vh;
  padding: 32px;
  background:
    linear-gradient(135deg, rgba(8, 47, 73, 0.88), rgba(15, 23, 42, 0.82)),
    url('@/assets/login-bg-industrial.png') center / cover;
}

.setup-card {
  width: min(560px, 100%);
  padding: 34px;
  border: 1px solid rgba(226, 238, 252, 0.22);
  border-radius: 18px;
  background: rgba(248, 251, 255, 0.92);
  box-shadow: 0 28px 70px rgba(8, 24, 48, 0.32);
}

.setup-card__header {
  margin-bottom: 24px;
}

.setup-card__tag {
  display: inline-flex;
  margin-bottom: 10px;
  color: var(--brand-strong);
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.setup-card h1 {
  margin: 0 0 8px;
  color: var(--text-primary);
  font-size: 28px;
  font-weight: 800;
  line-height: 1.2;
  letter-spacing: 0;
}

.setup-card p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 14px;
  line-height: 1.7;
}

.setup-form {
  display: grid;
  gap: 2px;
}

.setup-status {
  display: grid;
  gap: 14px;
  padding: 24px;
  border: 1px solid rgba(16, 50, 64, 0.12);
  border-radius: var(--radius-md);
  background: rgba(255, 255, 255, 0.68);
  color: var(--text-secondary);
}

.setup-status strong {
  color: var(--text-primary);
  font-size: 17px;
}

.setup-status--ready {
  justify-items: start;
}

.setup-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 10px;
}

:deep(.el-form-item) {
  margin-bottom: 16px;
}

:deep(.el-form-item__label) {
  color: var(--text-primary);
  font-weight: 700;
}

:deep(.el-input__wrapper) {
  border-radius: var(--radius-md);
  box-shadow: none;
}

@media (max-width: 640px) {
  .setup-page {
    padding: 20px;
  }

  .setup-card {
    padding: 28px 22px;
  }

  .setup-actions {
    display: grid;
  }
}
</style>
