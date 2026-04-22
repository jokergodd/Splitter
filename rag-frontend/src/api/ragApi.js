/**
 * RAG 知识库助手 API 接口层
 * 封装所有与后端交互的请求
 */

import axios from 'axios';

// 后端 API 基础地址，可根据实际部署修改
const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * 发送问答请求
 * @param {string} question - 用户问题
 * @param {string} sessionId - 会话ID（可选，用于多轮对话）
 * @returns {Promise<{answer: string, sources: Array}>}
 */
export async function askQuestion(question, sessionId = null) {
  // TODO: 替换为实际后端接口
  // const response = await api.post('/api/chat', { question, session_id: sessionId });
  // return response.data;

  // 模拟延迟响应（原型阶段）
  await new Promise((resolve) => setTimeout(resolve, 1500));
  return {
    answer: `这是关于「${question}」的回答。\n\n在实际对接后端后，这里会返回基于知识库检索生成的答案，并附带引用来源。`,
    sources: [
      { fileName: 'example.pdf', page: 3, score: 0.95 },
      { fileName: 'readme.md', score: 0.87 },
    ],
    sessionId: sessionId || 'mock-session-' + Date.now(),
  };
}

/**
 * 上传文件到知识库
 * @param {File} file - 文件对象
 * @param {Function} onProgress - 进度回调 (progress: number)
 * @returns {Promise<{fileId: string, fileName: string, status: string}>}
 */
export async function uploadFile(file, onProgress) {
  // TODO: 替换为实际后端接口
  // const formData = new FormData();
  // formData.append('file', file);
  // const response = await api.post('/api/upload', formData, {
  //   headers: { 'Content-Type': 'multipart/form-data' },
  //   onUploadProgress: (e) => {
  //     const progress = Math.round((e.loaded * 100) / e.total);
  //     onProgress?.(progress);
  //   },
  // });
  // return response.data;

  // 模拟上传（原型阶段）
  for (let i = 0; i <= 100; i += 10) {
    await new Promise((r) => setTimeout(r, 150));
    onProgress?.(i);
  }
  return {
    fileId: 'file-' + Date.now(),
    fileName: file.name,
    status: 'success',
    size: file.size,
    uploadedAt: new Date().toISOString(),
  };
}

/**
 * 获取知识库文件列表
 * @returns {Promise<Array<{fileId, fileName, size, uploadedAt, status}>>}
 */
export async function getFileList() {
  // TODO: 替换为实际后端接口
  // const response = await api.get('/api/files');
  // return response.data;

  return [];
}

/**
 * 删除知识库文件
 * @param {string} fileId - 文件ID
 */
export async function deleteFile(fileId) {
  // TODO: 替换为实际后端接口
  // await api.delete(`/api/files/${fileId}`);
  return { success: true };
}

/**
 * 获取文件处理状态
 * @param {string} fileId - 文件ID
 */
export async function getFileStatus(fileId) {
  // TODO: 替换为实际后端接口
  return { fileId, status: 'processed' };
}
