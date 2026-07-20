const API_BASE = '/api'

export async function chatStream(
  message: string,
  sessionId: string,
  onToken: (token: string) => void,
  onError: (error: string) => void,
  onDone: () => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  })

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = line.slice(6).trim()
        if (!data) continue

        try {
          const parsed = JSON.parse(data)
          if (parsed.type === 'token') {
            onToken(parsed.content)
          } else if (parsed.type === 'error') {
            onError(parsed.content || '未知错误')
          } else if (parsed.type === 'done') {
            onDone()
          }
        } catch {
          // skip malformed JSON
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

export async function uploadDocuments(files: File[]): Promise<string> {
  const formData = new FormData()
  files.forEach((f) => formData.append('files', f))

  const response = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: '上传失败' }))
    throw new Error(err.detail || `HTTP ${response.status}`)
  }

  const result = await response.json()
  return result.message
}

export async function clearDocuments(): Promise<void> {
  await fetch(`${API_BASE}/documents/clear`, { method: 'POST' })
}

export async function resetChat(): Promise<void> {
  await fetch(`${API_BASE}/reset`, { method: 'POST' })
}

export async function checkHealth(): Promise<{ status: string; vectorstore_loaded: boolean }> {
  const response = await fetch(`${API_BASE}/health`)
  return response.json()
}

export async function newSession(): Promise<string> {
  const response = await fetch(`${API_BASE}/chat/new`, { method: 'POST' })
  const data = await response.json()
  return data.session_id
}
