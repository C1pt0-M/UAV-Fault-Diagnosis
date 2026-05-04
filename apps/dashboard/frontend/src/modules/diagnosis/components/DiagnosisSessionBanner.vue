<template>
  <section v-if="activeSessionId" class="section-card">
    <div class="section-card__body">
      <el-alert
        type="warning"
        :closable="false"
        show-icon
        :title="diagnosisCopy.sessionBannerTitle"
        :description="diagnosisCopy.sessionBannerDescription(activeSessionId)"
      />
    </div>
  </section>
</template>

<script setup>
import ElAlert from 'element-plus/es/components/alert/index.mjs'
import { diagnosisCopy } from '@/modules/diagnosis/diagnosisView'

defineProps({
  activeSessionId: {
    type: String,
    default: ''
  }
})
</script>
