import { useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
}

export function DropZone({ onFile }: Props) {
  const [isDragOver, setIsDragOver] = useState(false);
  const dragCounter = useRef(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const onDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current++;
    setIsDragOver(true);
  };

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current--;
    if (dragCounter.current === 0) setIsDragOver(false);
  };

  const onDragOver = (e: React.DragEvent) => e.preventDefault();

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current = 0;
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFile(file);
  };

  return (
    <div
      className={`drop-zone${isDragOver ? " drag-over" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <div className="drop-content">
        <div className="drop-icon">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M9 18V5l12-2v13" />
            <circle cx="6" cy="18" r="3" />
            <circle cx="18" cy="16" r="3" />
          </svg>
        </div>
        <span className="drop-title">Drop an audio file here</span>
        <span className="drop-hint">Supports most audio formats — or click to browse</span>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        onChange={onInputChange}
        hidden
      />
    </div>
  );
}
