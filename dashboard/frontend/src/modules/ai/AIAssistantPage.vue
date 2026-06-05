<template>
  <div class="page-stack">
    <AIHeroSection />

    <section class="three-column">
      <AIConversationListPanel
        :conversations="conversations"
        :current-conversation-id="currentConversationId"
        :loading="loadingConversations"
        @open-conversation="openConversation"
        @start-fresh="startFreshConversation"
      />

      <AIChatPanel
        v-model:composer="composer"
        :loading-messages="loadingMessages"
        :messages="messages"
        :sending="sending"
        :can-seed-analysis="canSeedAnalysis"
        @send="sendMessage"
        @seed-analysis="seedReportAnalysis"
      />

      <AIContextPanel
        v-model:scene="scene"
        v-model:report-id-input="reportIdInput"
        :context-summary="contextSummary"
        :knowledge="knowledge"
        :reindexing="reindexingKnowledge"
        @reindex-knowledge="reindexKnowledge"
      />
    </section>
  </div>
</template>

<script setup>
import AIChatPanel from '@/modules/ai/components/AIChatPanel.vue'
import AIContextPanel from '@/modules/ai/components/AIContextPanel.vue'
import AIConversationListPanel from '@/modules/ai/components/AIConversationListPanel.vue'
import AIHeroSection from '@/modules/ai/components/AIHeroSection.vue'
import { useAIAssistantWorkspace } from '@/modules/ai/composables/useAIAssistantWorkspace'

const {
  canSeedAnalysis,
  composer,
  contextSummary,
  conversations,
  currentConversationId,
  knowledge,
  loadingConversations,
  loadingMessages,
  messages,
  openConversation,
  reportIdInput,
  reindexKnowledge,
  reindexingKnowledge,
  scene,
  seedReportAnalysis,
  sendMessage,
  sending,
  startFreshConversation
} = useAIAssistantWorkspace()
</script>

<style scoped>
.three-column {
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr) 320px;
  gap: 18px;
}
</style>
