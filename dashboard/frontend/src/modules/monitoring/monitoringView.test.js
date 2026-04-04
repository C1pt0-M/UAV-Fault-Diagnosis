import { describe, expect, it } from 'vitest'

import {
  resolveMonitoringBackendOptions,
  shouldShowBlindTest
} from '@/modules/monitoring/monitoringView'

describe('monitoring deployment capability helpers', () => {
  it('always keeps both backend options visible', () => {
    const options = resolveMonitoringBackendOptions({
      allowed_backends: ['replay'],
      blind_test_enabled: false,
      px4_enabled: false
    })

    expect(options).toEqual([
      { label: 'PX4 SITL', value: 'px4' },
      { label: 'Replay CSV', value: 'replay' }
    ])
  })

  it('returns default backend options when capability list is absent', () => {
    const options = resolveMonitoringBackendOptions({})

    expect(options).toEqual([
      { label: 'PX4 SITL', value: 'px4' },
      { label: 'Replay CSV', value: 'replay' }
    ])
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
