<template>
  <div class="page-stack ai-workspace">
    <AIContextPanel
      class="ai-layout__scene"
      panel-variant="scene"
      v-model:scene="scene"
      v-model:report-id-input="reportIdInput"
      :context-summary="contextSummary"
      :knowledge="knowledge"
      :reindexing="reindexingKnowledge"
      @reindex-knowledge="reindexKnowledge"
    />

    <section class="ai-layout">
      <AIConversationListPanel
        class="ai-layout__list"
        :rail-expanded="conversationRailExpanded"
        :conversations="conversations"
        :current-conversation-id="currentConversationId"
        :loading="loadingConversations"
        @open-conversation="openConversation"
        @start-fresh="startFreshConversation"
        @toggle-rail="toggleConversationRail"
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
        class="ai-layout__status"
        panel-variant="status"
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
import { ref } from 'vue'
import AIChatPanel from '@/modules/ai/components/AIChatPanel.vue'
import AIContextPanel from '@/modules/ai/components/AIContextPanel.vue'
import AIConversationListPanel from '@/modules/ai/components/AIConversationListPanel.vue'
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

const conversationRailExpanded = ref(false)

function toggleConversationRail() {
  conversationRailExpanded.value = !conversationRailExpanded.value
}
</script>

<style scoped>
.ai-workspace {
  --ai-side-width: 260px;
  --ai-rail-width: 56px;
  --ai-status-width: 300px;
}

.ai-layout__scene {
  position: sticky;
  top: var(--space-md);
  z-index: 2;
}

.ai-layout {
  display: grid;
  grid-template-columns: var(--ai-side-width) minmax(520px, 1fr) var(--ai-status-width);
  gap: var(--space-lg);
  align-items: start;
}

@media (max-width: 1560px) {
  .ai-layout {
    position: relative;
    grid-template-columns: var(--ai-rail-width) minmax(0, 1fr) var(--ai-status-width);
    grid-template-areas:
      "list chat context";
  }

  .ai-layout__chat {
    grid-area: chat;
  }

  .ai-layout__list {
    grid-area: list;
    position: sticky;
    top: calc(var(--space-md) + 92px);
    z-index: 3;
    align-self: start;
  }

  .ai-layout__status {
    grid-area: context;
  }

  .ai-layout__list.ai-list--rail-expanded {
    position: absolute;
    top: 0;
    left: 0;
    width: var(--ai-side-width);
    z-index: 5;
    box-shadow: var(--shadow-lg);
  }
}

@media (max-width: 1080px) {
  .ai-layout {
    grid-template-columns: 1fr;
    grid-template-areas:
      "chat"
      "context"
      "list";
  }

  .ai-layout__list,
  .ai-layout__list.ai-list--rail-expanded {
    position: static;
    width: auto;
    box-shadow: none;
  }
}
</style>
