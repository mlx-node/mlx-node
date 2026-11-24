import { execFile, type ExecFileOptions } from 'node:child_process';
import { win32 } from 'node:path';

/** How long one move to the Trash may take before the call gives up. */
const TRASH_TIMEOUT_MS = 60_000;

/**
 * Windows Recycle Bin script using Win32 COM `IFileOperation` + `IFileOperationProgressSink`.
 * Aborts with `E_ABORT` in `PreDeleteItem` if `TSF_DELETE_RECYCLE_IF_POSSIBLE` (`0x00000080`) is unset so deletion
 * never silently falls back to permanent removal. The target path is passed exclusively via `MLX_TRASH_PATH` (never
 * interpolated).
 */
const WIN_TRASH_SCRIPT = [
  "$ErrorActionPreference = 'Stop'",
  "Add-Type -TypeDefinition @'",
  'using System;',
  'using System.Runtime.InteropServices;',
  '',
  'namespace MlxTrash {',
  '  [ComImport]',
  '  [ComVisible(true)]',
  '  [Guid("04b0f1a7-9490-44bc-96e1-4296a31252e2")]',
  '  [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]',
  '  public interface IFileOperationProgressSink {',
  '    [PreserveSig] int StartOperations();',
  '    [PreserveSig] int FinishOperations(int hrResult);',
  '    [PreserveSig] int PreRenameItem(uint dwFlags, IntPtr psiItem, IntPtr pszNewName);',
  '    [PreserveSig] int PostRenameItem(uint dwFlags, IntPtr psiItem, IntPtr pszNewName, int hrRename, IntPtr psiNewlyCreated);',
  '    [PreserveSig] int PreMoveItem(uint dwFlags, IntPtr psiItem, IntPtr psiDestinationFolder, IntPtr pszNewName);',
  '    [PreserveSig] int PostMoveItem(uint dwFlags, IntPtr psiItem, IntPtr psiDestinationFolder, IntPtr pszNewName, int hrMove, IntPtr psiNewlyCreated);',
  '    [PreserveSig] int PreCopyItem(uint dwFlags, IntPtr psiItem, IntPtr psiDestinationFolder, IntPtr pszNewName);',
  '    [PreserveSig] int PostCopyItem(uint dwFlags, IntPtr psiItem, IntPtr psiDestinationFolder, IntPtr pszNewName, int hrCopy, IntPtr psiNewlyCreated);',
  '    [PreserveSig] int PreDeleteItem(uint dwFlags, IntPtr psiItem);',
  '    [PreserveSig] int PostDeleteItem(uint dwFlags, IntPtr psiItem, int hrDelete, IntPtr psiNewlyCreated);',
  '    [PreserveSig] int PreNewItem(uint dwFlags, IntPtr psiDestinationFolder, IntPtr pszNewName);',
  '    [PreserveSig] int PostNewItem(uint dwFlags, IntPtr psiDestinationFolder, IntPtr pszNewName, IntPtr pszTemplateName, uint dwFileAttributes, int hrNew, IntPtr psiNewItem);',
  '    [PreserveSig] int UpdateProgress(uint iWorkTotal, uint iWorkSoFar);',
  '    [PreserveSig] int ResetTimer();',
  '    [PreserveSig] int PauseTimer();',
  '    [PreserveSig] int ResumeTimer();',
  '  }',
  '',
  '  [ComImport]',
  '  [ComVisible(true)]',
  '  [Guid("947aab5f-0a5c-4c13-b4d6-4bf7836fc9f8")]',
  '  [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]',
  '  public interface IFileOperation {',
  '    [PreserveSig] int Advise(IFileOperationProgressSink pfops, out uint pdwCookie);',
  '    [PreserveSig] int Unadvise(uint dwCookie);',
  '    [PreserveSig] int SetOperationFlags(uint dwOperationFlags);',
  '    [PreserveSig] int SetProgressMessage([MarshalAs(UnmanagedType.LPWStr)] string pszMessage);',
  '    [PreserveSig] int SetProgressDialog(IntPtr popd);',
  '    [PreserveSig] int SetProperties(IntPtr pproparray);',
  '    [PreserveSig] int SetOwnerWindow(IntPtr hwndOwner);',
  '    [PreserveSig] int ApplyPropertiesToItem(IntPtr psiItem);',
  '    [PreserveSig] int ApplyPropertiesToItems(IntPtr punkItems);',
  '    [PreserveSig] int RenameItem(IntPtr psiItem, [MarshalAs(UnmanagedType.LPWStr)] string pszNewName, IFileOperationProgressSink pfopsItem);',
  '    [PreserveSig] int RenameItems(IntPtr pUnkItems, [MarshalAs(UnmanagedType.LPWStr)] string pszNewName);',
  '    [PreserveSig] int MoveItem(IntPtr psiItem, IntPtr psiDestinationFolder, [MarshalAs(UnmanagedType.LPWStr)] string pszNewName, IFileOperationProgressSink pfopsItem);',
  '    [PreserveSig] int MoveItems(IntPtr punkItems, IntPtr psiDestinationFolder);',
  '    [PreserveSig] int CopyItem(IntPtr psiItem, IntPtr psiDestinationFolder, [MarshalAs(UnmanagedType.LPWStr)] string pszCopyName, IFileOperationProgressSink pfopsItem);',
  '    [PreserveSig] int CopyItems(IntPtr punkItems, IntPtr psiDestinationFolder);',
  '    [PreserveSig] int DeleteItem(IntPtr psiItem, IFileOperationProgressSink pfopsItem);',
  '    [PreserveSig] int DeleteItems(IntPtr punkItems);',
  '    [PreserveSig] int NewItem(IntPtr psiDestinationFolder, uint dwFileAttributes, [MarshalAs(UnmanagedType.LPWStr)] string pszName, [MarshalAs(UnmanagedType.LPWStr)] string pszTemplateName, IFileOperationProgressSink pfopsItem);',
  '    [PreserveSig] int PerformOperations();',
  '    [PreserveSig] int GetAnyOperationsAborted([MarshalAs(UnmanagedType.Bool)] out bool pfAnyOperationsAborted);',
  '  }',
  '',
  '  [ComVisible(true)]',
  '  public sealed class RecycleOnlySink : IFileOperationProgressSink {',
  '    public bool SawPreDelete;',
  '    public bool SawRecycledPostDelete;',
  '    public int PostDeleteHr;',
  '    public int StartOperations() { return 0; }',
  '    public int FinishOperations(int hrResult) { return 0; }',
  '    public int PreRenameItem(uint dwFlags, IntPtr psiItem, IntPtr pszNewName) { return 0; }',
  '    public int PostRenameItem(uint dwFlags, IntPtr psiItem, IntPtr pszNewName, int hrRename, IntPtr psiNewlyCreated) { return 0; }',
  '    public int PreMoveItem(uint dwFlags, IntPtr psiItem, IntPtr psiDestinationFolder, IntPtr pszNewName) { return 0; }',
  '    public int PostMoveItem(uint dwFlags, IntPtr psiItem, IntPtr psiDestinationFolder, IntPtr pszNewName, int hrMove, IntPtr psiNewlyCreated) { return 0; }',
  '    public int PreCopyItem(uint dwFlags, IntPtr psiItem, IntPtr psiDestinationFolder, IntPtr pszNewName) { return 0; }',
  '    public int PostCopyItem(uint dwFlags, IntPtr psiItem, IntPtr psiDestinationFolder, IntPtr pszNewName, int hrCopy, IntPtr psiNewlyCreated) { return 0; }',
  '    public int PreDeleteItem(uint dwFlags, IntPtr psiItem) {',
  '      SawPreDelete = true;',
  '      if ((dwFlags & 0x00000080u) == 0) return unchecked((int)0x80004004);',
  '      return 0;',
  '    }',
  '    public int PostDeleteItem(uint dwFlags, IntPtr psiItem, int hrDelete, IntPtr psiNewlyCreated) {',
  '      PostDeleteHr = hrDelete;',
  '      if (hrDelete >= 0 && psiNewlyCreated != IntPtr.Zero) {',
  '        SawRecycledPostDelete = true;',
  '      }',
  '      return 0;',
  '    }',
  '    public int PreNewItem(uint dwFlags, IntPtr psiDestinationFolder, IntPtr pszNewName) { return 0; }',
  '    public int PostNewItem(uint dwFlags, IntPtr psiDestinationFolder, IntPtr pszNewName, IntPtr pszTemplateName, uint dwFileAttributes, int hrNew, IntPtr psiNewItem) { return 0; }',
  '    public int UpdateProgress(uint iWorkTotal, uint iWorkSoFar) { return 0; }',
  '    public int ResetTimer() { return 0; }',
  '    public int PauseTimer() { return 0; }',
  '    public int ResumeTimer() { return 0; }',
  '  }',
  '',
  '  public static class RecycleBin {',
  '    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = true)]',
  '    private static extern int SHCreateItemFromParsingName(',
  '      [MarshalAs(UnmanagedType.LPWStr)] string pszPath,',
  '      IntPtr pbc,',
  '      ref Guid riid,',
  '      out IntPtr ppv',
  '    );',
  '',
  '    public static void Move(string path) {',
  '      if (string.IsNullOrEmpty(path)) throw new ArgumentException("Trash path is empty.");',
  '      IntPtr shellItem = IntPtr.Zero;',
  '      IFileOperation op = null;',
  '      try {',
  '        Guid shellItemIid = new Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe");',
  '        int hr = SHCreateItemFromParsingName(path, IntPtr.Zero, ref shellItemIid, out shellItem);',
  '        if (hr < 0) Marshal.ThrowExceptionForHR(hr);',
  '        Type opType = Type.GetTypeFromCLSID(new Guid("3ad05575-8857-4850-9277-11b85bdb8e09"), true);',
  '        op = (IFileOperation)Activator.CreateInstance(opType);',
  '        uint flags = 0x0614u | 0x0040u | 0x00080000u | 0x20000000u | 0x00100000u;',
  '        hr = op.SetOperationFlags(flags);',
  '        if (hr < 0) Marshal.ThrowExceptionForHR(hr);',
  '        RecycleOnlySink sink = new RecycleOnlySink();',
  '        hr = op.DeleteItem(shellItem, sink);',
  '        if (hr < 0) Marshal.ThrowExceptionForHR(hr);',
  '        hr = op.PerformOperations();',
  '        if (hr < 0) Marshal.ThrowExceptionForHR(hr);',
  '        bool aborted = true;',
  '        hr = op.GetAnyOperationsAborted(out aborted);',
  '        if (hr < 0) Marshal.ThrowExceptionForHR(hr);',
  '        if (!sink.SawPreDelete || !sink.SawRecycledPostDelete || sink.PostDeleteHr < 0 || aborted) {',
  '          throw new InvalidOperationException("Recycle Bin operation was aborted or did not confirm a recycled item.");',
  '        }',
  '      } finally {',
  '        if (shellItem != IntPtr.Zero) Marshal.Release(shellItem);',
  '        if (op != null) Marshal.ReleaseComObject(op);',
  '      }',
  '    }',
  '  }',
  '}',
  "'@",
  '[MlxTrash.RecycleBin]::Move($env:MLX_TRASH_PATH)',
].join('\n');

function runExecFile(file: string, args: string[], options: ExecFileOptions): Promise<void> {
  return new Promise((resolve, reject) => {
    execFile(file, args, options, (error) => {
      if (error === null) resolve();
      else reject(error);
    });
  });
}

/**
 * Move `path` to the system Trash / Recycle Bin as one item: `/usr/bin/trash` on macOS, PowerShell (`-STA`) +
 * Win32 COM `IFileOperation` with a recycle-only `IFileOperationProgressSink` on Windows, and `gio trash` (falling
 * back to `trash-put` on ENOENT) on Linux/other platforms. Rejects when the tool cannot start or exits with an error.
 * The exit code alone is not proof: a caller checks the path afterwards.
 */
export async function moveToSystemTrash(path: string, platform: NodeJS.Platform = process.platform): Promise<void> {
  if (path.includes('\0')) {
    throw new Error('Trash path cannot contain NUL bytes.');
  }
  if (platform === 'darwin') {
    await runExecFile('/usr/bin/trash', ['-s', path], { timeout: TRASH_TIMEOUT_MS });
    return;
  }
  if (platform === 'win32') {
    const systemRoot = process.env.SystemRoot || process.env.SYSTEMROOT || 'C:\\Windows';
    const powershell = win32.join(systemRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe');
    await runExecFile(powershell, ['-STA', '-NoProfile', '-NonInteractive', '-Command', WIN_TRASH_SCRIPT], {
      timeout: TRASH_TIMEOUT_MS,
      windowsHide: true,
      env: { ...process.env, MLX_TRASH_PATH: path },
    });
    return;
  }
  try {
    await runExecFile('gio', ['trash', '--', path], { timeout: TRASH_TIMEOUT_MS });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    try {
      await runExecFile('trash-put', ['--', path], { timeout: TRASH_TIMEOUT_MS });
    } catch (fallbackError) {
      if ((fallbackError as NodeJS.ErrnoException).code === 'ENOENT') {
        throw new Error('No system trash command found (tried "gio" and "trash-put"). Install glib2 or trash-cli.');
      }
      throw fallbackError;
    }
  }
}
