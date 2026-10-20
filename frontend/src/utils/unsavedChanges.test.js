// Run from frontend/: node --test src/utils/unsavedChanges.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { hasUnsavedChanges, unsavedFilePaths, renameBlockedMessage } from './unsavedChanges.js';

const saved = { path: 'a.py', name: 'a.py', code: 'x = 1\n', savedCode: 'x = 1\n' };
const edited = { path: 'b.py', name: 'b.py', code: 'y = 2\n', savedCode: '' };

test('a file whose editor content matches the last-saved content is not unsaved', () => {
  assert.equal(hasUnsavedChanges(saved), false);
});

test('a file edited after it was opened or saved has unsaved changes', () => {
  assert.equal(hasUnsavedChanges(edited), true);
  assert.equal(hasUnsavedChanges({ ...saved, code: 'x = 2\n' }), true);
});

test('saving (savedCode updated to the editor content) clears the unsaved state', () => {
  assert.equal(hasUnsavedChanges({ ...edited, savedCode: edited.code }), false);
});

test('a file without a recorded saved state is not treated as unsaved', () => {
  assert.equal(hasUnsavedChanges({ path: 'c.py', code: 'z' }), false);
});

test('only files with unsaved changes are listed', () => {
  assert.deepEqual(unsavedFilePaths([saved, edited]), ['b.py']);
  assert.deepEqual(unsavedFilePaths([]), []);
});

test('rename is allowed when no open file has unsaved changes', () => {
  assert.equal(renameBlockedMessage([saved]), null);
  assert.equal(renameBlockedMessage([]), null);
});

test('rename is blocked with a message naming every file with unsaved changes', () => {
  const other = { ...edited, path: 'Webpage/main.py', name: 'main.py' };
  const message = renameBlockedMessage([saved, edited, other]);
  assert.match(message, /unsaved changes/);
  assert.match(message, /- b\.py/);
  assert.match(message, /- Webpage\/main\.py/);
  assert.doesNotMatch(message, /- a\.py/);
});
