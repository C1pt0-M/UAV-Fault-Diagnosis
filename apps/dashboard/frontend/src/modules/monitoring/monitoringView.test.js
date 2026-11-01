import { describe, expect, it } from 'vitest'

import {
  formatMonitoringModelName,
  monitoringDurationOptions,
  monitoringStateLabels,
  resolveMonitoringBackendOptions,
  shouldShowBlindTest
} from '@/modules/monitoring/monitoringView'

describe('monitoring deployment capability helpers', () => {
  it('does not offer a backend outside the deployment allowlist', () => {
    const options = resolveMonitoringBackendOptions({
      enabled: true,
      allowed_backends: ['replay'],
      blind_test_enabled: false,
      px4_enabled: false
    })

    expect(options).toEqual([{ label: 'CSV 回放', value: 'replay' }])
  })

  it.each([{}, { enabled: false, allowed_backends: ['rfly'] }, { enabled: true, allowed_backends: 'replay' }])('offers no source for unavailable capability %j', (capabilities) => {
    expect(resolveMonitoringBackendOptions(capabilities)).toEqual([])
  })

  it('keeps allowed but not-ready RflySim visible without restoring hidden PX4', () => {
    expect(resolveMonitoringBackendOptions({ enabled: true, allowed_backends: ['rfly', 'px4'], rfly_ready: false }))
      .toEqual([{ label: 'RflySim 实时链路', value: 'rfly' }])
  })

  it('requires both the blind-test flag and allowed PX4 backend', () => {
    expect(shouldShowBlindTest({ enabled: true, allowed_backends: ['rfly'], blind_test_enabled: true })).toBe(false)
    expect(shouldShowBlindTest({ enabled: true, allowed_backends: ['px4'], blind_test_enabled: 'true' })).toBe(false)
    expect(shouldShowBlindTest({ enabled: true, allowed_backends: ['px4'], blind_test_enabled: true })).toBe(true)
  })

  it('uses readable Chinese labels for states and durations', () => {
    expect(monitoringStateLabels).toMatchObject({
      IDLE: '空闲',
      COLLECTING: '采集中',
      DETECTING: '检测中',
      BLIND_TEST: '盲测中'
    })
    expect(monitoringDurationOptions.map((item) => item.label)).toEqual(['30 秒', '60 秒', '120 秒'])
  })

  it('hides blind-test action when capability is disabled', () => {
    expect(
      shouldShowBlindTest({
        allowed_backends: ['replay'],
        blind_test_enabled: false
      })
    ).toBe(false)
  })

  it('formats online detector model names for compact KPI display', () => {
    expect(formatMonitoringModelName('online_detector_kg_mstca_rfly')).toBe('KG-MSTCA')
    expect(formatMonitoringModelName('KG-MSTCA 在线检测模型')).toBe('KG-MSTCA')
    expect(formatMonitoringModelName('online_detector_cnn1d_rfly')).toBe('CNN1D')
    expect(formatMonitoringModelName('DroneDiag-v2.3.1')).toBe('DroneDiag-v2.3.1')
  })
})
