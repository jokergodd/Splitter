import React, { useState, useRef } from 'react';
import { Input, Button, Tooltip, Badge } from 'antd';
import {
  SendOutlined,
  PaperClipOutlined,
  LoadingOutlined,
} from '@ant-design/icons';

const { TextArea } = Input;

/**
 * 聊天输入组件
 * 支持文本输入、文件上传快捷入口
 */
export default function ChatInput({
  onSend,
  onUploadClick,
  isLoading,
  uploadedFileCount,
}) {
  const [input, setInput] = useState('');
  const textAreaRef = useRef(null);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    onSend(trimmed);
    setInput('');
    // 保持焦点
    setTimeout(() => textAreaRef.current?.focus(), 100);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-input-area">
      <div className="chat-input-toolbar">
        <Tooltip title="上传知识库文件">
          <Button
            icon={
              uploadedFileCount > 0 ? (
                <Badge count={uploadedFileCount} size="small" offset={[2, -2]}>
                  <PaperClipOutlined />
                </Badge>
              ) : (
                <PaperClipOutlined />
              )
            }
            onClick={onUploadClick}
            className="toolbar-btn"
          >
            知识库
          </Button>
        </Tooltip>
        {uploadedFileCount > 0 && (
          <span className="file-hint">已加载 {uploadedFileCount} 个文件</span>
        )}
      </div>
      <div className="chat-input-box">
        <TextArea
          ref={textAreaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入问题，AI 将基于知识库为你解答..."
          autoSize={{ minRows: 1, maxRows: 6 }}
          disabled={isLoading}
          className="chat-textarea"
        />
        <Button
          type="primary"
          icon={isLoading ? <LoadingOutlined /> : <SendOutlined />}
          onClick={handleSend}
          disabled={!input.trim() || isLoading}
          className="send-btn"
        />
      </div>
      <div className="chat-input-footer">
        <span className="footer-hint">Shift + Enter 换行</span>
      </div>
    </div>
  );
}
