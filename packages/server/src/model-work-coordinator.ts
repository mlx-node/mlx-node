/**
 * Process-local gate for native MLX work.
 *
 * Individual model instances already have a per-model execution mutex, but a
 * lazy `loadModel()` can still run load-time materialization / warmup Metal
 * work while another model is decoding. MLX's allocator and command queues are
 * process-wide, so model load/swap takes an exclusive writer slot; inference
 * takes shared reader slots.
 */
export class ModelWorkCoordinator {
  private activeReaders = 0;
  private writerActive = false;
  private waitingWriters = 0;
  private readonly readerWaiters: Array<() => void> = [];
  private readonly writerWaiters: Array<() => void> = [];

  async withModelLoad<T>(fn: () => Promise<T> | T): Promise<T> {
    await this.acquireWrite();
    try {
      return await fn();
    } finally {
      this.releaseWrite();
    }
  }

  async withInference<T>(fn: () => Promise<T> | T): Promise<T> {
    await this.acquireRead();
    try {
      return await fn();
    } finally {
      this.releaseRead();
    }
  }

  private acquireRead(): Promise<void> {
    if (!this.writerActive && this.waitingWriters === 0) {
      this.activeReaders += 1;
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      this.readerWaiters.push(() => {
        this.activeReaders += 1;
        resolve();
      });
    });
  }

  private acquireWrite(): Promise<void> {
    this.waitingWriters += 1;
    if (!this.writerActive && this.activeReaders === 0) {
      this.waitingWriters -= 1;
      this.writerActive = true;
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      this.writerWaiters.push(() => {
        this.waitingWriters -= 1;
        this.writerActive = true;
        resolve();
      });
    });
  }

  private releaseRead(): void {
    this.activeReaders -= 1;
    if (this.activeReaders < 0) this.activeReaders = 0;
    if (this.activeReaders === 0) this.drain();
  }

  private releaseWrite(): void {
    this.writerActive = false;
    this.drain();
  }

  private drain(): void {
    if (this.writerActive) return;
    if (this.activeReaders === 0 && this.writerWaiters.length > 0) {
      this.writerWaiters.shift()?.();
      return;
    }
    if (this.waitingWriters === 0 && this.readerWaiters.length > 0) {
      const readers = this.readerWaiters.splice(0);
      for (const resolve of readers) resolve();
    }
  }
}
