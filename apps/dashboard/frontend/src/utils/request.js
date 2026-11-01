import { clearStoredSession } from '@/shared/utils/user'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

function resolveHealthUrl(apiBaseUrl) {
  if (!apiBaseUrl) return '/health'

  if (apiBaseUrl.startsWith('http://') || apiBaseUrl.startsWith('https://')) {
    try {
      const url = new URL(apiBaseUrl)
      return `${url.origin}/health`
    } catch {
      return '/health'
    }
  }

  return '/health'
}

const HEALTH_URL = resolveHealthUrl(API_BASE_URL)

let connectionStatus = 'unknown'
let connectionCallbacks = []

function notifyConnectionChange(status) {
  connectionStatus = status
  connectionCallbacks.forEach((callback) => {
    try {
      callback(status)
    } catch (error) {
      console.error('连接状态回调执行失败:', error)
    }
  })
}

export function onConnectionChange(callback) {
  connectionCallbacks.push(callback)
  callback(connectionStatus)
}

export function offConnectionChange(callback) {
  connectionCallbacks = connectionCallbacks.filter((item) => item !== callback)
}

export function getConnectionStatus() {
  return connectionStatus
}

async function doRequest(path, options = {}) {
  const {
    params = null,
    data = null,
    formData = null,
    headers = {},
    method = 'GET',
    responseType = 'json'
  } = options

  const token = localStorage.getItem('token')
  const finalHeaders = { ...headers }

  if (token) {
    finalHeaders.Authorization = `Bearer ${token}`
  }

  let url = `${API_BASE_URL}${path}`

  if (params) {
    const query = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        query.append(key, value)
      }
    })

    const queryString = query.toString()
    if (queryString) {
      url += `?${queryString}`
    }
  }

  const fetchOptions = {
    method,
    headers: finalHeaders
  }

  if (formData) {
    fetchOptions.body = formData
  } else if (data !== null && data !== undefined) {
    fetchOptions.body = JSON.stringify(data)
    finalHeaders['Content-Type'] = finalHeaders['Content-Type'] || 'application/json'
  }

  try {
    const response = await fetch(url, fetchOptions)

    if (response.status === 401) {
      if (localStorage.getItem('token') === token) {
        clearStoredSession()
        if (window.location.pathname !== '/login') {
          window.location.href = '/login'
        }
      }
      throw new Error('未登录或登录已过期')
    }

    if (!response.ok) {
      let message = `HTTP ${response.status}`
      let errorData = null

      try {
        errorData = await response.json()
        message = errorData.message || message
      } catch {
        // 保留默认 HTTP 错误信息。
      }

      const error = new Error(message)
      error.status = response.status
      error.data = errorData
      throw error
    }

    if (connectionStatus !== 'connected') {
      notifyConnectionChange('connected')
    }

    if (responseType === 'blob') return response.blob()
    if (responseType === 'text') return response.text()

    const json = await response.json()
    if (json.success === false) {
      throw new Error(json.message || '请求失败')
    }
    return json
  } catch (error) {
    if (
      error.message.includes('Failed to fetch') ||
      error.message.includes('NetworkError') ||
      error.name === 'AbortError'
    ) {
      notifyConnectionChange('disconnected')
    }
    throw error
  }
}

export async function checkBackend() {
  notifyConnectionChange('checking')
  try {
    const response = await fetch(HEALTH_URL, { method: 'GET' })
    if (response.ok) {
      notifyConnectionChange('connected')
      return true
    }
    notifyConnectionChange('disconnected')
    return false
  } catch {
    notifyConnectionChange('disconnected')
    return false
  }
}

export async function waitForBackend(maxAttempts = 10, interval = 2000) {
  for (let i = 0; i < maxAttempts; i += 1) {
    if (await checkBackend()) {
      return true
    }
    if (i < maxAttempts - 1) {
      await new Promise((resolve) => setTimeout(resolve, interval))
    }
  }
  return false
}

export default {
  get(url, options = {}) {
    return doRequest(url, { ...options, method: 'GET' })
  },
  post(url, data, options = {}) {
    return doRequest(url, { ...options, method: 'POST', data })
  },
  put(url, data, options = {}) {
    return doRequest(url, { ...options, method: 'PUT', data })
  },
  delete(url, options = {}) {
    return doRequest(url, { ...options, method: 'DELETE' })
  },
  postForm(url, formData, options = {}) {
    return doRequest(url, { ...options, method: 'POST', formData })
  },
  getBlob(url, options = {}) {
    return doRequest(url, { ...options, method: 'GET', responseType: 'blob' })
  }
}
