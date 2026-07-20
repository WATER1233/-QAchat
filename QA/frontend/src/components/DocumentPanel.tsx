import { useRef } from 'react'
import { useDocuments } from '../hooks/useDocuments'

interface Props {
  isUploading: boolean
  status: string
  isLoaded: boolean
  onUpload: (files: File[]) => void
  onClear: () => void
}

export default function DocumentPanel({ isUploading, status, isLoaded, onUpload, onClear }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      onUpload(Array.from(files))
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const files = Array.from(e.dataTransfer.files).filter((f) => f.name.endsWith('.pdf'))
    if (files.length > 0) onUpload(files)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-5 py-4 border-b border-gray-100">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-primary-100 flex items-center justify-center">
            <svg className="w-4 h-4 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <span className="font-semibold text-sm text-gray-800">文档管理</span>
        </div>
      </div>

      <div className="flex-1 px-5 py-4 space-y-4 overflow-y-auto">
        {/* Status */}
        {status && (
          <div className={`px-3 py-2.5 rounded-lg text-xs leading-relaxed ${
            status.startsWith('已加载') || status.startsWith('已清空')
              ? status.startsWith('已加载')
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                : 'bg-gray-50 text-gray-500 border border-gray-200'
              : status.startsWith('加载失败') || status.startsWith('请选择')
                ? 'bg-amber-50 text-amber-700 border border-amber-200'
                : 'bg-blue-50 text-blue-600 border border-blue-200'
          }`}>
            {status}
          </div>
        )}

        {/* Upload area */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => inputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-200 ${
            isUploading
              ? 'border-primary-300 bg-primary-50/50'
              : 'border-gray-200 hover:border-primary-300 hover:bg-primary-50/30'
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf"
            multiple
            onChange={handleFileChange}
            className="hidden"
          />
          {isUploading ? (
            <div className="flex flex-col items-center gap-2">
              <div className="w-8 h-8 border-2 border-primary-400 border-t-transparent rounded-full animate-spin" />
              <span className="text-xs text-primary-600 font-medium">正在处理…</span>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <svg className="w-8 h-8 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <span className="text-xs text-gray-400">
                点击选择或拖拽 PDF 到此处
              </span>
              <span className="text-[11px] text-gray-300">支持多文件上传</span>
            </div>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex gap-2">
          <button
            onClick={() => inputRef.current?.click()}
            disabled={isUploading}
            className="flex-1 px-3 py-2 bg-primary-600 text-white text-xs font-medium rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isUploading ? '处理中…' : '加载文档'}
          </button>
          {isLoaded && (
            <button
              onClick={onClear}
              className="px-3 py-2 border border-gray-200 text-gray-500 text-xs font-medium rounded-lg hover:bg-gray-50 transition-colors"
            >
              清空
            </button>
          )}
        </div>

        {/* Tips */}
        <div className="pt-2">
          <div className="flex items-center gap-1.5 mb-2">
            <svg className="w-3.5 h-3.5 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">使用提示</span>
          </div>
          <div className="space-y-1.5">
            {[
              ['📎', '可上传多个 PDF，自动建库统一检索'],
              ['💬', '支持文档问答和日常闲聊'],
              ['🔍', '回答标注来源文档和页码'],
            ].map(([icon, text]) => (
              <div key={text} className="flex items-start gap-2 text-[12px] text-gray-400 leading-relaxed">
                <span>{icon}</span>
                <span>{text}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
