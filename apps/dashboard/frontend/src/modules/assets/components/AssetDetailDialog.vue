<template>
  <el-dialog v-model="dialogVisible" :title="assetsCopy.detailTitle" width="720px">
    <pre class="asset-json">{{ detailJson }}</pre>
  </el-dialog>
</template>

<script setup>
import { computed } from 'vue'
import ElDialog from 'element-plus/es/components/dialog/index.mjs'
import { assetsCopy } from '@/modules/assets/assetsView'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  detailJson: {
    type: String,
    default: '{}'
  }
})

const emit = defineEmits(['update:visible'])

const dialogVisible = computed({
  get: () => props.visible,
  set: (value) => emit('update:visible', value)
})
</script>

<style scoped>
.asset-json {
  margin: 0;
  max-height: 420px;
  overflow: auto;
  padding: 16px;
  border-radius: var(--radius-md);
  border: 1px solid var(--line-soft);
  background: var(--bg-panel-strong);
  color: var(--text-primary);
}
</style>
