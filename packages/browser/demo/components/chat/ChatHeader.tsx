export type ChatHeaderProps = {
  onReset: () => void;
};

export function ChatHeader({ onReset }: ChatHeaderProps) {
  return (
    <div className="chat-header">
      <div className="chat-avatar">Q</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <div className="chat-header-title">Qwen 3.5 Vision</div>
        <div className="chat-header-status">Ready on WebGPU</div>
      </div>
      <div style={{ marginLeft: 'auto' }}>
        <button type="button" className="btn-reset" onClick={onReset}>
          Reset Chat
        </button>
      </div>
    </div>
  );
}
