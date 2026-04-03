import { describe, expect, it } from 'vitest'

import {
  resolveMonitoringBackendOptions,
  shouldShowBlindTest
} from '@/modules/monitoring/monitoringView'

describe('monitoring deployment capability helpers', () => {
  it('filters backend options to replay when replay-only capability is returned', () => {
    const options = resolveMonitoringBackendOptions({
      allowed_backends: ['replay'],
      blind_test_enabled: false,
      px4_enabled: false
    })

    expect(options).toEqual([{ label: 'Replay CSV', value: 'replay' }])
  })

  it('hides blind-test action when capability is disabled', () => {
    expect(
      shouldShowBlindTest({
        allowed_backends: ['replay'],
        blind_test_enabled: false
      })
    ).toBe(false)
  })
})
