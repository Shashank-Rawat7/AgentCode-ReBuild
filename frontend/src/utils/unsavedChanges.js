// Unsaved-change tracking for open editor tabs.
// Each open file keeps `code` (what is in the editor) and `savedCode`
// (what was last read from or saved to disk).

export function hasUnsavedChanges(file) {
  return file.savedCode !== undefined && file.code !== file.savedCode;
}

export function unsavedFilePaths(openFiles) {
  return openFiles.filter(hasUnsavedChanges).map(f => f.path);
}

// Renaming reloads the page, which discards every open tab's editor content,
// so a rename is refused while any open file has unsaved changes.
export function renameBlockedMessage(openFiles) {
  const paths = unsavedFilePaths(openFiles);
  if (paths.length === 0) return null;
  return (
    'Rename cancelled: these files have unsaved changes:\n' +
    paths.map(p => `  - ${p}`).join('\n') +
    '\n\nSave them first (Save or Ctrl+S). Renaming reloads the page and would discard unsaved changes.'
  );
}
