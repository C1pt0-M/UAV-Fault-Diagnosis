import { describe, expect, it } from 'vitest'

import {
  resolveMonitoringBackendOptions,
  shouldShowBlindTest
} from '@/modules/monitoring/monitoringView'

describe('monitoring deployment capability helpers', () => {
  it('keeps RflySim visible even when backend capability is not ready', () => {
    const options = resolveMonitoringBackendOptions({
      allowed_backends: ['replay'],
      blind_test_enabled: false,
      px4_enabled: false
    })

    expect(options).toEqual([
      { label: 'RflySim', value: 'rfly' },
      { label: 'CSV 回放', value: 'replay' }
    ])
  })

  it('returns default backend options when capability list is absent', () => {
    const options = resolveMonitoringBackendOptions({})

    expect(options).toEqual([
      { label: 'RflySim', value: 'rfly' },
      { label: 'CSV 回放', value: 'replay' }
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
