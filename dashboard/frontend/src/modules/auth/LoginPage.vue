<template>
  <div class="login-page">
    <!-- 全屏背景层 -->
    <div class="login-bg-layer" :style="{ backgroundImage: 'url(' + loginBgImage + ')' }">
      <div class="login-bg-overlay"></div>
    </div>

    <!-- 悬浮表单容器 -->
    <div class="login-card-container">
      <section class="login-card">
        <div class="login-header">
          <h1>{{ isRegisterMode ? '创建新账号' : '登录系统' }}</h1>
          <p>{{ isRegisterMode ? '注册后可直接返回登录' : '欢迎登录智诊天眸' }}</p>
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
          <el-button type="primary" class="login-button" :loading="submitting" @click="submit">
            {{ isRegisterMode ? '创建账号' : '登录系统' }}
          </el-button>
        </el-form>

        <div class="login-footer">
          <span>{{ isRegisterMode ? '已有账号？' : '还没有账号？' }}</span>
          <el-button link type="primary" @click="isRegisterMode ? switchToLogin() : switchToRegister()">
            {{ isRegisterMode ? '返回登录' : '立即注册' }}
          </el-button>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import { ElForm, ElFormItem } from 'element-plus/es/components/form/index.mjs'
import ElInput from 'element-plus/es/components/input/index.mjs'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { authApi } from '@/shared/services'
import loginBgImage from '@/assets/login-bg-full.png'

const router = useRouter()
const formRef = ref(null)
const submitting = ref(false)
const mode = ref('login')
const isRegisterMode = computed(() => mode.value === 'register')

const form = reactive({
  username: '',
  password: ''
})

const rules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
}

function clearPassword() {
  form.password = ''
}

function switchToRegister() {
  mode.value = 'register'
  clearPassword()
}

function switchToLogin() {
  mode.value = 'login'
  clearPassword()
}

async function submit() {
  if (!formRef.value) return

  await formRef.value.validate(async (valid) => {
    if (!valid) return

    submitting.value = true
    try {
      if (isRegisterMode.value) {
        await authApi.register({
          username: form.username,
          password: form.password
        })
        ElMessage.success('注册成功，请使用新账号登录')
        switchToLogin()
        return
      }

      const response = await authApi.login({
        username: form.username,
        password: form.password
      })

      localStorage.setItem('token', response.data.token)
      localStorage.setItem('user', JSON.stringify(response.data.user))
      ElMessage.success('登录成功')
      router.push('/')
    } catch (error) {
      ElMessage.error(
        error.message || (isRegisterMode.value ? '注册失败，请检查输入信息' : '登录失败，请检查账号和密码')
      )
    } finally {
      submitting.value = false
    }
  })
}
</script>

<style scoped>
.login-page {
  position: relative;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: flex-start;
  overflow: hidden;
}

/* ── 全屏背景层 ── */
.login-bg-layer {
  position: absolute;
  inset: 0;
  background-size: cover;
  background-position: center;
  z-index: 0;
}

.login-bg-overlay {
  position: absolute;
  inset: 0;
  background: rgba(15, 23, 42, 0.35); /* 稍微降低一点浓度，凸显背景 */
  backdrop-filter: blur(2px);
}

/* ── 悬浮卡片 ── */
.login-card-container {
  position: relative;
  z-index: 1;
  width: 100%;
  max-width: 340px; /* 极致缩小，让出更多背景空间 */
  margin-left: 12vw;
  display: flex;
  justify-content: center;
}

.login-card {
  width: 100%;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  padding: 36px 28px; /* 压缩内边距 */
  border-radius: 16px;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.login-header {
  margin-bottom: 24px;
}

.login-header h1 {
  font-size: 24px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0 0 6px 0;
}

.login-header p {
  color: var(--text-tertiary);
  font-size: 12px;
  margin: 0;
}

.login-button {
  width: 100%;
  margin-top: 16px;
  height: 40px;
  font-size: 14px;
  border-radius: var(--radius-md);
}

.login-footer {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  margin-top: 24px;
  color: var(--text-tertiary);
  font-size: 13px;
}

@media (max-width: 900px) {
  .login-page {
    justify-content: center;
  }
  .login-card-container {
    margin-left: 0;
    padding: 20px;
  }
}
</style>
