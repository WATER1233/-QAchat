import { useState, useCallback } from 'react'
import { uploadDocuments, clearDocuments } from '../services/api'

export function useDocuments() {
  const [status, setStatus] = useState<string>('请上传 PDF 文件开始问答')
  const [isUploading, setIsUploading] = useState(false)
  const [isLoaded, setIsLoaded] = useState(false)

  const upload = useCallback(async (files: File[]) => {
    if (!files.length) return

    setIsUploading(true)
    setStatus('正在解析文档并生成向量库，请稍候…')

    try {
      const message = await uploadDocuments(files)
      setStatus(message)
      setIsLoaded(true)
    } catch (err: any) {
      setStatus(`加载失败: ${err.message}`)
    } finally {
      setIsUploading(false)
    }
  }, [])

  const clear = useCallback(async () => {
    try {
      await clearDocuments()
      setIsLoaded(false)
      setStatus('已清空文档索引')
    } catch {
      setStatus('清空失败')
    }
  }, [])

  return {
    status,
    isUploading,
    isLoaded,
    upload,
    clear,
  }
}
