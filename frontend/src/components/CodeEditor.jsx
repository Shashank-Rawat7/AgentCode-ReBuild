
/* === CodeEditor.jsx === */
import Editor from '@monaco-editor/react';
import { useRef } from 'react';

export default function CodeEditor({
  language,
  code,
  onCodeChange,
  onCursorChange,
  onLanguageChange,
  children,
}) {
  const editorRef = useRef(null);

  return (
    <div
      style={{
        height: '100%',
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'flex-end',
          padding: '6px 8px',
          background: '#1e1e1e',
        }}
      >
        <select
          aria-label="Programming language"
          value={language}
          onChange={(event) => onLanguageChange(event.target.value)}
          style={{
            background: '#2d2d2d',
            color: '#ffffff',
            border: '1px solid #555555',
            borderRadius: '4px',
            padding: '6px 8px',
          }}
        >
          {children}
        </select>
      </div>

      <div style={{ flex: 1, minHeight: 0, width: '100%', position: 'relative' }}>
        <Editor
          height="100%"
          width="100%"
          language={language}
          theme="vs-dark"
          value={code}
          onChange={onCodeChange}
          onMount={(editor) => {
            editorRef.current = editor;
            setTimeout(() => editor.layout(), 50);

            editor.onDidChangeCursorPosition(() => {
              const pos = editor.getPosition();
              if (pos && onCursorChange) {
                onCursorChange({
                  line: pos.lineNumber,
                  column: pos.column,
                });
              }
            });

            const pos = editor.getPosition();
            if (pos && onCursorChange) {
              onCursorChange({
                line: pos.lineNumber,
                column: pos.column,
              });
            }
          }}
          options={{ fontSize: 14, minimap: { enabled: false } }}
        />
      </div>
    </div>
  );
}
