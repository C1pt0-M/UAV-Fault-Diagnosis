<template>
  <div class="setup-page">
    <section class="setup-card">
      <span class="setup-card__tag">初始化入口</span>
      <h1>系统已切换到新的前端工作台</h1>
      <p>
        当前版本优先提供稳定的登录入口与核心业务闭环。如果需要继续使用系统，
        直接进入登录页即可。
      </p>
      <el-button type="primary" size="large" @click="router.push('/login')">进入登录页</el-button>
    </section>
  </div>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
import { useRouter } from 'vue-router'

const router = useRouter()
</script>

<style scoped>
.setup-page {
  display: grid;
  place-items: center;
  min-height: 100vh;
  padding: 32px;
}

.setup-card {
  width: min(560px, 100%);
  padding: 36px;
  border-radius: 28px;
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(16, 50, 64, 0.12);
  box-shadow: 0 24px 60px rgba(15, 31, 38, 0.12);
}

.setup-card__tag {
  display: inline-flex;
  padding: 6px 10px;
  border-radius: 999px;
  background: var(--brand-soft);
  color: var(--brand-strong);
  font-size: 12px;
  font-weight: 700;
}

.setup-card h1 {
  margin: 14px 0 0;
  font-size: 30px;
}

.setup-card p {
  margin: 14px 0 22px;
  color: var(--text-secondary);
  line-height: 1.8;
}
</style>
