import {
  ChatDotRound,
  DataAnalysis,
  Document,
  House,
  Odometer
} from '@element-plus/icons-vue'

export const navigationItems = [
  {
    key: 'overview',
    label: '总览',
    path: '/',
    icon: House,
    matchPaths: ['/']
  },
  {
    key: 'monitoring',
    label: '实时监测',
    path: '/monitoring',
    icon: Odometer,
    matchPaths: ['/monitoring']
  },
  {
    key: 'diagnosis',
    label: '诊断中心',
    path: '/diagnosis',
    icon: DataAnalysis,
    matchPaths: ['/diagnosis']
  },
  {
    key: 'workspace',
    label: '资产与报告',
    path: '/assets',
    icon: Document,
    matchPaths: ['/assets', '/reports']
  },
  {
    key: 'ai',
    label: 'AI 助手',
    path: '/ai',
    icon: ChatDotRound,
    matchPaths: ['/ai']
  }
]

export const pageMeta = {
  '/': {
    title: '总览',
    description: '查看系统总体状态、离线诊断统计、最近任务与风险分布。'
  },
  '/monitoring': {
    title: '实时监测',
    description: '统一管理遥测采集、在线检测、实时告警与会话历史。'
  },
  '/diagnosis': {
    title: '诊断中心',
    description: '执行离线诊断任务，联动会话追溯与诊断报告生成。'
  },
  '/assets': {
    title: '资产中心',
    description: '管理 ULG / CSV 日志资产，完成上传、预处理、下载与追踪。'
  },
  '/reports': {
    title: '诊断报告',
    description: '查看、筛选与下载离线诊断报告，连接报告与资产上下文。'
  },
  '/ai': {
    title: 'AI 助手',
    description: '围绕报告、任务和知识库进行问答分析与原因解释。'
  }
}

export function resolvePageMeta(path) {
  return pageMeta[path] || {
    title: '工作台',
    description: '围绕无人机故障检测闭环开展日常分析与运维。'
  }
}
