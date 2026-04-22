import React, { useState } from 'react';
import {
  Upload,
  Button,
  List,
  Tag,
  Progress,
  Empty,
  Tooltip,
  Popconfirm,
  Typography,
  message,
} from 'antd';
import {
  InboxOutlined,
  FilePdfOutlined,
  FileMarkdownOutlined,
  FileWordOutlined,
  FileTextOutlined,
  DeleteOutlined,
  CloudUploadOutlined,
  FileUnknownOutlined,
} from '@ant-design/icons';
import { uploadFile, deleteFile } from '../api/ragApi';

const { Dragger } = Upload;
const { Text, Title } = Typography;

/**
 * 获取文件类型图标
 */
function getFileIcon(fileName) {
  const ext = fileName.split('.').pop()?.toLowerCase();
  switch (ext) {
    case 'pdf':
      return <FilePdfOutlined style={{ color: '#ff4d4f', fontSize: 20 }} />;
    case 'md':
    case 'markdown':
      return <FileMarkdownOutlined style={{ color: '#1677ff', fontSize: 20 }} />;
    case 'doc':
    case 'docx':
      return <FileWordOutlined style={{ color: '#2b579a', fontSize: 20 }} />;
    case 'txt':
      return <FileTextOutlined style={{ color: '#52c41a', fontSize: 20 }} />;
    default:
      return <FileUnknownOutlined style={{ color: '#8c8c8c', fontSize: 20 }} />;
  }
}

/**
 * 获取文件类型标签
 */
function getFileTypeTag(fileName) {
  const ext = fileName.split('.').pop()?.toLowerCase();
  const typeMap = {
    pdf: { color: 'red', text: 'PDF' },
    md: { color: 'blue', text: 'MD' },
    markdown: { color: 'blue', text: 'MD' },
    doc: { color: 'processing', text: 'WORD' },
    docx: { color: 'processing', text: 'WORD' },
    txt: { color: 'green', text: 'TXT' },
  };
  const info = typeMap[ext] || { color: 'default', text: ext?.toUpperCase() || 'FILE' };
  return <Tag color={info.color}>{info.text}</Tag>;
}

/**
 * 格式化文件大小
 */
function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

/**
 * 左侧边栏组件
 * 包含文件上传和知识库文件列表
 */
export default function Sidebar({ files, onFilesChange }) {
  const [uploadingMap, setUploadingMap] = useState({});

  const handleUpload = async (file) => {
    // 检查文件类型
    const allowedExts = ['pdf', 'md', 'markdown', 'doc', 'docx', 'txt'];
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!allowedExts.includes(ext)) {
      message.error(`不支持的文件类型: .${ext}，请上传 PDF/Markdown/Word/TXT 文件`);
      return false;
    }

    // 检查是否已存在
    if (files.some((f) => f.fileName === file.name)) {
      message.warning(`文件「${file.name}」已存在`);
      return false;
    }

    const uploadId = Date.now() + '-' + file.name;

    // 开始上传
    setUploadingMap((prev) => ({
      ...prev,
      [uploadId]: { fileName: file.name, progress: 0 },
    }));

    try {
      const result = await uploadFile(file, (progress) => {
        setUploadingMap((prev) => ({
          ...prev,
          [uploadId]: { ...prev[uploadId], progress },
        }));
      });

      onFilesChange((prev) => [...prev, result]);
      message.success(`「${file.name}」上传成功`);
    } catch (error) {
      message.error(`上传失败: ${error.message}`);
    } finally {
      setUploadingMap((prev) => {
        const next = { ...prev };
        delete next[uploadId];
        return next;
      });
    }

    return false; // 阻止 antd 默认上传行为
  };

  const handleDelete = async (fileId) => {
    try {
      await deleteFile(fileId);
      onFilesChange((prev) => prev.filter((f) => f.fileId !== fileId));
      message.success('文件已删除');
    } catch (error) {
      message.error('删除失败');
    }
  };

  const uploadingList = Object.values(uploadingMap);

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <Title level={5} className="sidebar-title">
          <CloudUploadOutlined /> 知识库文件
        </Title>
        <Text type="secondary" className="sidebar-subtitle">
          上传文档供 AI 检索回答
        </Text>
      </div>

      {/* 上传区域 */}
      <div className="sidebar-upload">
        <Dragger
          beforeUpload={handleUpload}
          showUploadList={false}
          accept=".pdf,.md,.markdown,.doc,.docx,.txt"
          multiple
        >
          <p className="upload-icon">
            <InboxOutlined />
          </p>
          <p className="upload-text">点击或拖拽文件上传</p>
          <p className="upload-hint">
            支持 PDF、Markdown、Word、TXT
          </p>
        </Dragger>
      </div>

      {/* 上传进度 */}
      {uploadingList.length > 0 && (
        <div className="upload-progress-area">
          <Text type="secondary" strong>上传中...</Text>
          {uploadingList.map((item) => (
            <div key={item.fileName} className="upload-progress-item">
              <Text ellipsis style={{ maxWidth: 180 }}>
                {getFileIcon(item.fileName)} {item.fileName}
              </Text>
              <Progress
                percent={item.progress}
                size="small"
                status={item.progress < 100 ? 'active' : 'success'}
              />
            </div>
          ))}
        </div>
      )}

      {/* 文件列表 */}
      <div className="file-list-area">
        <div className="file-list-header">
          <Text strong>已上传文件 ({files.length})</Text>
        </div>
        {files.length === 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="暂无文件"
            className="file-empty"
          />
        ) : (
          <List
            className="file-list"
            dataSource={files}
            renderItem={(file) => (
              <List.Item
                className="file-list-item"
                actions={[
                  <Popconfirm
                    key="delete"
                    title="确认删除"
                    description={`删除「${file.fileName}」后，AI 将不再引用此文件内容`}
                    onConfirm={() => handleDelete(file.fileId)}
                    okText="删除"
                    cancelText="取消"
                  >
                    <Tooltip title="删除">
                      <Button
                        type="text"
                        danger
                        icon={<DeleteOutlined />}
                        size="small"
                      />
                    </Tooltip>
                  </Popconfirm>,
                ]}
              >
                <List.Item.Meta
                  avatar={getFileIcon(file.fileName)}
                  title={
                    <Text ellipsis style={{ maxWidth: 160 }} title={file.fileName}>
                      {file.fileName}
                    </Text>
                  }
                  description={
                    <div className="file-meta">
                      {getFileTypeTag(file.fileName)}
                      <Text type="secondary">{formatSize(file.size)}</Text>
                    </div>
                  }
                />
              </List.Item>
            )}
          />
        )}
      </div>
    </aside>
  );
}
