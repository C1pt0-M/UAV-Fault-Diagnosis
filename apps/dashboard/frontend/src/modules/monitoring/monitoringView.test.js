import { describe, expect, it } from 'vitest'

import {
  formatMonitoringModelName,
  monitoringDurationOptions,
  monitoringStateLabels,
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
      { label: 'RflySim 实时链路', value: 'rfly' },
      { label: 'CSV 回放', value: 'replay' }
    ])
  })

  it('returns default backend options when capability list is absent', () => {
    const options = resolveMonitoringBackendOptions({})

    expect(options).toEqual([
      { label: 'RflySim 实时链路', value: 'rfly' },
      { label: 'CSV 回放', value: 'replay' }
    ])
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
