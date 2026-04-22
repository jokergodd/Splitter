import React, { useRef, useEffect } from 'react';
import { Empty, Typography, Button } from 'antd';
import { RobotOutlined, ClearOutlined } from '@ant-design/icons';
import MessageBubble from './MessageBubble';
import ChatInput from './ChatInput';

const { Text } = Typography;

/**
 * 聊天区域组件
 * 展示消息列表和输入框
 */
export default function ChatArea({
  messages,
  isLoading,
  onSend,
  onClear,
  onUploadClick,
  uploadedFileCount,
}) {
  const messagesEndRef = useRef(null);

  // 自动滚动到底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <main className="chat-area">
      {/* 顶部导航栏 */}
      <header className="chat-header">
        <div className="chat-header-left">
          <RobotOutlined className="header-icon" />
          <div className="header-info">
            <Text strong className="header-title">
              RAG 知识库助手
            </Text>
            <Text type="secondary" className="header-status">
              {isLoading ? '思考中...' : '就绪'}
            </Text>
          </div>
        </div>
        {messages.length > 0 && (
          <Button
            icon={<ClearOutlined />}
            onClick={onClear}
            size="small"
            className="clear-btn"
          >
            清空对话
          </Button>
        )}
      </header>

      {/* 消息列表 */}
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <div className="welcome-icon">
              <RobotOutlined />
            </div>
            <Text strong className="welcome-title">
              欢迎使用 RAG 知识库助手
            </Text>
            <Text type="secondary" className="welcome-desc">
              在左侧上传 PDF、Markdown、Word 或 TXT 文件，<br />
              然后在这里提问，AI 会基于你的知识库给出精准回答。
            </Text>
            <div className="welcome-examples">
              <div className="example-tag">📄 支持 PDF 文档解析</div>
              <div className="example-tag">📝 支持 Markdown 格式</div>
              <div className="example-tag">📘 支持 Word 文档</div>
              <div className="example-tag">📃 支持 TXT 文本</div>
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* 输入区域 */}
      <ChatInput
        onSend={onSend}
        onUploadClick={onUploadClick}
        isLoading={isLoading}
        uploadedFileCount={uploadedFileCount}
      />
    </main>
  );
}
