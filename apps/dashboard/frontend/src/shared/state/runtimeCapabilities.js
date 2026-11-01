import { reactive, readonly } from 'vue'

const state = reactive({ ai: false, telemetry: false })

// In-memory deployment capabilities, refreshed by authenticated profile checks.
// Missing or malformed declarations never enable an optional page.
export const runtimeCapabilities = readonly(state)

export function applyRuntimeCapabilities(capabilities) {
  state.ai = capabilities?.ai === true
  state.telemetry = capabilities?.telemetry === true
}

export function resetRuntimeCapabilities() {
  applyRuntimeCapabilities(null)
}
