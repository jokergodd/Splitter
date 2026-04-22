import React from 'react';
import { Avatar, Tag, Typography } from 'antd';
import {
  UserOutlined,
  RobotOutlined,
  FileTextOutlined,
} from '@ant-design/icons';

const { Text } = Typography;

/**
 * 消息气泡组件
 * 展示用户提问和 AI 回答
 */
export default function MessageBubble({ message }) {
  const { role, content, sources, timestamp, isLoading } = message;
  const isUser = role === 'user';

  return (
    <div className={`message-row ${isUser ? 'message-user' : 'message-ai'}`}>
      <div className="message-avatar">
        <Avatar
          size={36}
          icon={isUser ? <UserOutlined /> : <RobotOutlined />}
          style={{
            backgroundColor: isUser ? '#1677ff' : '#52c41a',
          }}
        />
      </div>
      <div className="message-content-wrapper">
        <div className="message-header">
          <Text strong>{isUser ? '你' : 'AI 助手'}</Text>
          {timestamp && (
            <Text type="secondary" className="message-time">
              {new Date(timestamp).toLocaleTimeString()}
            </Text>
          )}
        </div>
        <div className="message-bubble">
          {isLoading ? (
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          ) : (
            <div className="message-text">{content}</div>
          )}
        </div>

        {/* 引用来源 */}
        {!isUser && sources && sources.length > 0 && (
          <div className="message-sources">
            <Text type="secondary" className="sources-label">
              <FileTextOutlined /> 引用来源：
            </Text>
            <div className="sources-list">
              {sources.map((source, idx) => (
                <Tag key={idx} color="blue" className="source-tag">
                  {source.fileName}
                  {source.page && ` (P${source.page})`}
                </Tag>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
