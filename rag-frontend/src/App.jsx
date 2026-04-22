import { useState, useCallback } from 'react';
import { ConfigProvider, App as AntApp } from 'antd';
import { MenuFoldOutlined, MenuUnfoldOutlined } from '@ant-design/icons';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import { askQuestion } from './api/ragApi';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [files, setFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // 发送消息
  const handleSend = useCallback(async (content) => {
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content,
      timestamp: Date.now(),
    };

    const loadingMessage = {
      id: Date.now() + 1,
      role: 'assistant',
      content: '',
      isLoading: true,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage, loadingMessage]);
    setIsLoading(true);

    try {
      const result = await askQuestion(content);

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingMessage.id
            ? {
                ...msg,
                isLoading: false,
                content: result.answer,
                sources: result.sources,
              }
            : msg
        )
      );
    } catch (error) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingMessage.id
            ? {
                ...msg,
                isLoading: false,
                content: '抱歉，请求出现了错误，请稍后重试。',
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  // 清空对话
  const handleClear = useCallback(() => {
    setMessages([]);
  }, []);

  // 切换侧边栏
  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev) => !prev);
  }, []);

  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1677ff',
          borderRadius: 8,
        },
      }}
    >
      <AntApp>
        <div className="app-container">
          {/* 移动端侧边栏切换按钮 */}
          <button className="sidebar-toggle" onClick={toggleSidebar}>
            {sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
          </button>

          {/* 侧边栏 */}
          <div
            className={`sidebar-wrapper ${
              sidebarCollapsed ? 'sidebar-hidden' : ''
            }`}
          >
            <Sidebar files={files} onFilesChange={setFiles} />
          </div>

          {/* 遮罩层（移动端） */}
          {!sidebarCollapsed && (
            <div className="sidebar-overlay" onClick={toggleSidebar} />
          )}

          {/* 聊天区域 */}
          <div className="chat-wrapper">
            <ChatArea
              messages={messages}
              isLoading={isLoading}
              onSend={handleSend}
              onClear={handleClear}
              onUploadClick={() => setSidebarCollapsed(false)}
              uploadedFileCount={files.length}
            />
          </div>
        </div>
      </AntApp>
    </ConfigProvider>
  );
}

export default App;
