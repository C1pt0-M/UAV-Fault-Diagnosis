<template>
  <div class="login-page">
    <section class="login-card">
      <div class="login-card__hero">
        <span class="login-card__tag">注册入口</span>
        <h1>创建新账号</h1>
        <p>
          使用最小信息完成注册。创建成功后会自动跳转到登录页，使用新账号进入系统。
        </p>
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
          />
        </el-form-item>
        <el-form-item label="姓名" prop="real_name">
          <el-input v-model="form.real_name" placeholder="请输入姓名" />
        </el-form-item>
        <el-form-item label="邮箱" prop="email">
          <el-input v-model="form.email" placeholder="请输入邮箱" />
        </el-form-item>
        <el-button type="primary" size="large" class="login-button" :loading="submitting" @click="submit">
          创建账号
        </el-button>
      </el-form>

      <div class="auth-card__footer">
        <span>已有账号？</span>
        <el-button link type="primary" @click="router.push('/login')">返回登录</el-button>
      </div>
    </section>
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

const router = useRouter()
const formRef = ref(null)
const submitting = ref(false)

const form = reactive({
  username: '',
  password: '',
  real_name: '',
  email: ''
})

const rules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
  real_name: [{ required: true, message: '请输入姓名', trigger: 'blur' }],
  email: [{ required: true, message: '请输入邮箱', trigger: 'blur' }]
}

async function submit() {
  if (!formRef.value) return

  await formRef.value.validate(async (valid) => {
    if (!valid) return

    submitting.value = true
    try {
      await authApi.register({
        username: form.username,
        password: form.password,
        real_name: form.real_name,
        email: form.email
      })
      ElMessage.success('注册成功，请使用新账号登录')
      router.push('/login')
    } catch (error) {
      ElMessage.error(error.message || '注册失败，请检查输入信息')
    } finally {
      submitting.value = false
    }
  })
}
</script>

<style scoped>
.login-page {
  display: grid;
  place-items: center;
  min-height: 100vh;
  padding: 32px;
}

.login-card {
  width: min(520px, 100%);
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
  margin: 12px 0 24px;
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

.login-button {
  width: 100%;
  margin-top: 8px;
}

.auth-card__footer {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  margin-top: 18px;
  color: var(--text-tertiary);
  font-size: 13px;
}
</style>
