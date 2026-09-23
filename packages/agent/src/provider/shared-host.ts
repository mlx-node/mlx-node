import { ChatSession, PagedConfigOverrideManager, type SessionCapableModel } from '@mlx-node/lm';
import { ModelWorkCoordinator, SessionRegistry } from '@mlx-node/server';

import { MlxModelHost } from './model-host.js';
import { SHARED_REQUEST_LIMIT, SHARED_SESSION_LIMIT, type SharedProfile } from './shared-protocol.js';
import type { StreamSimpleHost } from './stream-adapter.js';

interface SharedResident {
  key: string;
  model: SessionCapableModel;
  admission: SessionRegistry;
  host: MlxModelHost;
  overlays: PagedConfigOverrideManager;
  sessions: Map<string, SharedSession>;
  preparing: Promise<void>;
}

interface SharedSession {
  session: ChatSession;
  busy: boolean;
  dirty: boolean;
}

/** One set of weights, with independent sessions on the model's native scheduler. */
export class SharedInferenceHost {
  private readonly coordinator = new ModelWorkCoordinator();
  private resident?: SharedResident;
  private readonly owners = new Map<string, Promise<void>>();
  pending = 0;

  constructor(
    private readonly makeHost = (profile: SharedProfile, overlays: PagedConfigOverrideManager) =>
      new MlxModelHost([profile.discovered], {
        requirePagedCache: true,
        persistPagedCache: profile.persistPagedCache,
        resolveModelPathFn: (model, policy) => overlays.resolve(model.path, model.modelType, policy?.persistPagedCache),
      }),
  ) {}

  forRequest(profile: SharedProfile, signal: AbortSignal): StreamSimpleHost {
    const key = JSON.stringify(profile);
    let active: SharedSession | undefined;
    const discard = () => {
      if (active) active.dirty = true;
    };
    return {
      modelInfo: (id) => (id === profile.discovered.name ? profile.discovered : undefined),
      // Failed turns are disposed before releasing this owner's admission lane.
      markResidentDirty: discard,
      consumeResidentDirty: () => false,
      invalidateResident: discard,
      runWithResident: async (_id, fn, owner) => {
        if (this.pending >= SHARED_REQUEST_LIMIT) throw new Error('Shared inference queue is full.');
        this.pending++;
        let loaded = false;
        const previous = owner ? this.owners.get(owner) : undefined;
        let release!: () => void;
        const lane = new Promise<void>((resolve) => {
          release = resolve;
        });
        if (owner) this.owners.set(owner, lane);
        try {
          // Two overlapping turns from the same client/session must never use
          // or dispose the same native owner concurrently. Other owners batch.
          await previous;
          for (;;) {
            signal.throwIfAborted();
            const outcome = await this.coordinator.withInference(async () => {
              if (this.resident?.key !== key) return undefined;
              const resident = this.resident;
              const run = async () => {
                // Serialize cache allocation/eviction, not inference. Otherwise a
                // resumed owner could race the asynchronous release of its old
                // idle session and lose the new turn's native state.
                const prepared = resident.preparing.then(async () => {
                  signal.throwIfAborted();
                  await resident.admission.flushPendingDisposals();
                  if (resident.admission.pendingDisposalCount > 0)
                    throw new Error('Could not release a previous inference session.');
                  const paged = resident.model.hasBlockPagedCache?.() === true;
                  let entry = owner ? resident.sessions.get(owner) : undefined;
                  if (!entry) {
                    // Bound retained owners; flat models can only keep one warm
                    // session because their native cache is model-global.
                    const limit = paged ? SHARED_SESSION_LIMIT : 1;
                    if (resident.sessions.size >= limit) {
                      const oldest = Array.from(resident.sessions).find(([, value]) => !value.busy);
                      if (!oldest) throw new Error('All shared inference sessions are busy.');
                      resident.sessions.delete(oldest[0]);
                      await resident.admission.disposeSession(oldest[1].session);
                    }
                    entry = { session: new ChatSession(resident.model), busy: false, dirty: false };
                    if (owner) resident.sessions.set(owner, entry);
                    if (!paged) {
                      try {
                        await entry.session.reset();
                      } catch (error) {
                        if (owner) resident.sessions.delete(owner);
                        await resident.admission.disposeSession(entry.session);
                        throw error;
                      }
                    }
                  }
                  entry.busy = true;
                  return entry;
                });
                resident.preparing = prepared.then(
                  () => {},
                  () => {},
                );
                const entry = await prepared;
                active = entry;
                let completed = false;
                try {
                  const value = await fn(entry.session, !loaded);
                  completed = true;
                  return { value };
                } finally {
                  entry.busy = false;
                  active = undefined;
                  if (owner) resident.sessions.delete(owner);
                  if (owner && completed && !entry.dirty && !signal.aborted) {
                    // Reinsertion keeps the least recently used idle owner first.
                    resident.sessions.set(owner, entry);
                  } else await resident.admission.disposeSession(entry.session);
                }
              };
              return resident.admission.concurrentAdmissionLimit > 1
                ? resident.admission.withAdmission(run)
                : resident.admission.withExclusive(run);
            });
            if (outcome) return outcome.value;
            await this.coordinator.withModelLoad(async () => {
              signal.throwIfAborted();
              if (this.resident?.key === key) return;
              await this.disposeResident();
              const overlays = new PagedConfigOverrideManager({
                preserveEmbeddedGemmaDraft: profile.preserveEmbeddedGemmaDraft,
              });
              const host = this.makeHost(profile, overlays);
              try {
                await host.runWithModel(profile.discovered.name, async (model) => {
                  const capacity = model.hasBlockPagedCache?.() === true ? model.maxConcurrentSequences?.() : 1;
                  this.resident = {
                    key,
                    model,
                    host,
                    overlays,
                    sessions: new Map(),
                    preparing: Promise.resolve(),
                    admission: new SessionRegistry({
                      model,
                      maxQueueDepth: 16,
                      maxConcurrentDispatches:
                        Number.isSafeInteger(capacity) && capacity! > 1 ? Math.min(capacity!, SHARED_SESSION_LIMIT) : 1,
                    }),
                  };
                  loaded = true;
                });
              } catch (error) {
                await host.dispose();
                await overlays.cleanup();
                throw error;
              }
            });
          }
        } finally {
          release();
          if (owner && this.owners.get(owner) === lane) this.owners.delete(owner);
          this.pending--;
        }
      },
    };
  }

  async close(): Promise<void> {
    await this.coordinator.withModelLoad(() => this.disposeResident());
  }

  private async disposeResident(): Promise<void> {
    const resident = this.resident;
    if (!resident) return;
    for (const [owner, entry] of resident.sessions) {
      resident.sessions.delete(owner);
      await resident.admission.disposeSession(entry.session);
    }
    await resident.admission.flushPendingDisposals();
    if (resident.admission.pendingDisposalCount > 0) throw new Error('Could not release the resident model sessions.');
    await resident.host.dispose();
    this.resident = undefined;
    await resident.overlays.cleanup();
  }
}
