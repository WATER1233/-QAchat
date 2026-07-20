export interface Message {
  role: 'user' | 'assistant'
  content: string
}

export interface SSEMessage {
  type: 'token' | 'done' | 'error'
  content?: string
}

export interface UploadResult {
  status: string
  message: string
}
