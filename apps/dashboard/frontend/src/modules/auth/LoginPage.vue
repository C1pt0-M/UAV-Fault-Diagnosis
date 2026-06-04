<template>
  <div class="login-page">
    <div class="login-bg-layer" :style="{ backgroundImage: 'url(' + loginBgImage + ')' }">
      <div class="login-bg-overlay"></div>
    </div>

    <div class="login-card-container">
      <section class="login-card">
        <div class="login-header">
          <span class="login-header__eyebrow">Research Platform</span>
          <h1>智诊天眸</h1>
          <p>无人机故障诊断研究平台</p>
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
            登录系统
          </el-button>
        </el-form>

        <div class="login-footer">
          <span>首次使用？</span>
          <el-button link type="primary" @click="router.push('/setup')">进入系统初始化</el-button>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import { ElForm, ElFormItem } from 'element-plus/es/components/form/index.mjs'
import ElInput from 'element-plus/es/components/input/index.mjs'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { authApi } from '@/shared/services'
import { storeSession } from '@/shared/utils/user'
import loginBgImage from '@/assets/login-bg-industrial.png'

const router = useRouter()
const formRef = ref(null)
const submitting = ref(false)

const form = reactive({
  username: '',
  password: ''
})

const rules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, max: 20, message: '密码长度需为 6-20 位', trigger: 'blur' }
  ]
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

      storeSession(response.data.token, response.data.user)
      ElMessage.success('登录成功')
      router.push('/')
    } catch (error) {
      ElMessage.error(error.message || '登录失败，请检查账号和密码')
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
  justify-content: center;
  overflow: hidden;
  padding: 24px;
}

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
  background:
    radial-gradient(circle at 50% 48%, rgba(226, 242, 255, 0.08), rgba(15, 23, 42, 0.3) 58%, rgba(15, 23, 42, 0.45)),
    linear-gradient(135deg, rgba(8, 47, 73, 0.28), rgba(15, 23, 42, 0.34));
  backdrop-filter: blur(1.5px);
}

.login-card-container {
  position: relative;
  z-index: 1;
  width: 100%;
  max-width: 410px;
  display: flex;
  justify-content: center;
}

.login-card {
  width: 100%;
  padding: 36px 32px 32px;
  border: 1px solid rgba(255, 255, 255, 0.34);
  border-radius: 18px;
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.28), rgba(241, 247, 255, 0.14));
  box-shadow: 0 28px 70px rgba(8, 24, 48, 0.34);
  backdrop-filter: blur(18px) saturate(140%);
  -webkit-backdrop-filter: blur(18px) saturate(140%);
  color: var(--text-primary);
}

.login-header {
  margin-bottom: 24px;
  text-align: left;
}

.login-header__eyebrow {
  display: inline-flex;
  margin-bottom: 10px;
  color: rgba(219, 234, 254, 0.88);
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.login-header h1 {
  margin: 0 0 8px;
  color: #f8fbff;
  font-size: 28px;
  font-weight: 800;
  line-height: 1.2;
  letter-spacing: 0;
  text-shadow: 0 1px 18px rgba(15, 23, 42, 0.24);
}

.login-header p {
  color: rgba(226, 238, 252, 0.88);
  font-size: 13px;
  margin: 0;
}

.login-field-hint {
  margin: 6px 0 0;
  color: rgba(226, 238, 252, 0.68);
  font-size: 12px;
  line-height: 1.5;
}

:deep(.el-form-item) {
  margin-bottom: 18px;
}

:deep(.el-form-item__label) {
  color: rgba(240, 247, 255, 0.92);
  font-weight: 700;
}

:deep(.el-input__wrapper) {
  border: 1px solid rgba(226, 238, 252, 0.24);
  border-radius: var(--radius-md);
  background: rgba(255, 255, 255, 0.2);
  box-shadow: none;
  backdrop-filter: blur(10px);
}

:deep(.el-input__wrapper.is-focus) {
  border-color: rgba(96, 165, 250, 0.82);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.18);
}

:deep(.el-input__inner) {
  color: #f8fbff;
}

:deep(.el-input__inner::placeholder) {
  color: rgba(226, 238, 252, 0.58);
}

.login-button {
  width: 100%;
  margin-top: 8px;
  height: 42px;
  font-size: 14px;
  border-radius: var(--radius-md);
  font-weight: 800;
  box-shadow: 0 12px 24px rgba(37, 99, 235, 0.28);
}

.login-footer {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  margin-top: 22px;
  color: rgba(226, 238, 252, 0.76);
  font-size: 13px;
}

:deep(.login-footer .el-button) {
  color: #bfdbfe;
  font-weight: 800;
}

@media (max-width: 900px) {
  .login-card-container {
    max-width: 390px;
  }

  .login-card {
    padding: 32px 24px 28px;
  }
}
</style>
