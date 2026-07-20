import { useState, useRef, useCallback, useEffect } from 'react'
import type { Message } from '../types'
import { chatStream, newSession as createSession } from '../services/api'

const STORAGE_KEY = 'pdfqa_session_id'

function loadSession(): string | null {
  return localStorage.getItem(STORAGE_KEY)
}

function saveSession(id: string) {
  localStorage.setItem(STORAGE_KEY, id)
}

function clearSession() {
  localStorage.removeItem(STORAGE_KEY)
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const sessionIdRef = useRef<string>('')

  // 初始化：尝试恢复旧 session，否则创建新 session
  useEffect(() => {
    const existing = loadSession()
    if (existing) {
      sessionIdRef.current = existing
      // 从后端恢复历史
      fetch(`/api/chat/${existing}`)
        .then((r) => r.json())
        .then((data) => {
          if (data.messages?.length) {
            setMessages(data.messages)
          }
        })
        .catch(() => {
          // 后端没找到 session，重建一个
          createSession().then((id) => {
            sessionIdRef.current = id
            saveSession(id)
          })
        })
    } else {
      createSession().then((id) => {
        sessionIdRef.current = id
        saveSession(id)
      })
    }
  }, [])

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return

    setError(null)
    setIsLoading(true)

    const userMessage: Message = { role: 'user', content }
    const assistantMessage: Message = { role: 'assistant', content: '' }

    setMessages((prev) => [...prev, userMessage, assistantMessage])

    const abortController = new AbortController()
    abortRef.current = abortController

    try {
      let fullContent = ''
      await chatStream(
        content,
        sessionIdRef.current,
        (token) => {
          fullContent += token
          setMessages((prev) => {
            const updated = [...prev]
            const last = updated[updated.length - 1]
            if (last && last.role === 'assistant') {
              updated[updated.length - 1] = { ...last, content: fullContent }
            }
            return updated
          })
        },
        (err) => {
          setError(err)
          setMessages((prev) => {
            const updated = [...prev]
            if (updated[updated.length - 1]?.role === 'assistant' && !updated[updated.length - 1].content) {
              updated.pop()
              if (updated[updated.length - 1]?.role === 'user') updated.pop()
            }
            return updated
          })
        },
        () => {
          // done
        },
        abortController.signal
      )
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setError(err.message || '请求失败')
        setMessages((prev) => {
          const updated = [...prev]
          if (updated[updated.length - 1]?.role === 'assistant' && !updated[updated.length - 1].content) {
            updated.pop()
            if (updated[updated.length - 1]?.role === 'user') updated.pop()
          }
          return updated
        })
      }
    } finally {
      setIsLoading(false)
      abortRef.current = null
    }
  }, [isLoading])

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const clearMessages = useCallback(async () => {
    setMessages([])
    setError(null)
    clearSession()
    const id = await createSession()
    sessionIdRef.current = id
    saveSession(id)
  }, [])

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    stopGeneration,
    clearMessages,
  }
}
