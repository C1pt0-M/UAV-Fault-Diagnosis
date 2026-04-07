<template>
  <div class="page-stack ai-workspace">
    <AIHeroSection />

    <section class="ai-layout">
      <AIConversationListPanel
        class="ai-layout__list"
        :conversations="conversations"
        :current-conversation-id="currentConversationId"
        :loading="loadingConversations"
        @open-conversation="openConversation"
        @start-fresh="startFreshConversation"
      />

      <AIChatPanel
        class="ai-layout__chat"
        v-model:composer="composer"
        :loading-messages="loadingMessages"
        :messages="messages"
        :sending="sending"
        :can-seed-analysis="canSeedAnalysis"
        @send="sendMessage"
        @seed-analysis="seedReportAnalysis"
      />

      <AIContextPanel
        class="ai-layout__context"
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
.ai-layout {
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr) 320px;
  gap: 18px;
  align-items: start;
}

@media (max-width: 1560px) {
  .ai-layout {
    grid-template-columns: minmax(0, 1fr) 320px;
    grid-template-areas:
      "chat context"
      "list context";
  }

  .ai-layout__chat {
    grid-area: chat;
  }

  .ai-layout__list {
    grid-area: list;
  }

  .ai-layout__context {
    grid-area: context;
  }
}

@media (max-width: 1320px) {
  .ai-layout {
    grid-template-columns: 1fr;
    grid-template-areas:
      "chat"
      "context"
      "list";
  }
}
</style>
