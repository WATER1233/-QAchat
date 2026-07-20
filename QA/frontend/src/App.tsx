import { useChat } from './hooks/useChat'
import { useDocuments } from './hooks/useDocuments'
import DocumentPanel from './components/DocumentPanel'
import ChatPanel from './components/ChatPanel'

export default function App() {
  const { messages, isLoading, error, sendMessage, stopGeneration, clearMessages } = useChat()
  const { status, isUploading, isLoaded, upload, clear } = useDocuments()

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Top bar */}
      <header className="flex-none bg-white border-b border-gray-200 px-6 py-3">
        <div className="max-w-7xl mx-auto flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary-600 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-bold text-gray-800">PDF 智能问答</h1>
              <span className="text-[10px] font-medium text-primary-600 bg-primary-50 px-1.5 py-0.5 rounded">v2</span>
            </div>
            <p className="text-[11px] text-gray-400">上传文档 · 智能解析 · 即刻问答</p>
          </div>
          {!isLoaded && (
            <div className="ml-auto flex items-center gap-1.5 text-[11px] text-amber-600 bg-amber-50 px-3 py-1.5 rounded-full border border-amber-200">
              <span className="w-1.5 h-1.5 bg-amber-400 rounded-full" />
              未加载文档
            </div>
          )}
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden max-w-7xl mx-auto w-full">
        {/* Left sidebar */}
        <aside className="w-[300px] flex-none bg-white border-r border-gray-200 overflow-y-auto">
          <DocumentPanel
            isUploading={isUploading}
            status={status}
            isLoaded={isLoaded}
            onUpload={upload}
            onClear={clear}
          />
        </aside>

        {/* Right chat area */}
        <main className="flex-1 flex flex-col bg-gray-50">
          <ChatPanel
            messages={messages}
            isLoading={isLoading}
            error={error}
            onSend={sendMessage}
            onStop={stopGeneration}
            onClear={clearMessages}
          />
        </main>
      </div>
    </div>
  )
}
