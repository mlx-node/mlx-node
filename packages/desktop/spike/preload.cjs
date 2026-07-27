// Sandboxed preloads are CJS. Sandbox stays ON — verified compatible with the
// MessagePort transfer the real app will use.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('spike', {
  onToken: (cb) => ipcRenderer.on('spike:token', (_e, text) => cb(text)),
});
