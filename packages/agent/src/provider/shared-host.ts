import { ChatSession, PagedConfigOverrideManager, type SessionCapableModel } from '@mlx-node/lm';
import { ModelWorkCoordinator, SessionRegistry } from '@mlx-node/server';

import { MlxModelHost } from './model-host.js';
import { SHARED_REQUEST_LIMIT, type SharedProfile } from './shared-protocol.js';
import type { StreamSimpleHost } from './stream-adapter.js';

interface SharedResident {
  key: string;
  model: SessionCapableModel;
  admission: SessionRegistry;
  host: MlxModelHost;
  overlays: PagedConfigOverrideManager;
}

/** One set of weights, with independent sessions on the model's native scheduler. */
export class SharedInferenceHost {
  private readonly coordinator = new ModelWorkCoordinator();
  private resident?: SharedResident;
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
    return {
      modelInfo: (id) => (id === profile.discovered.name ? profile.discovered : undefined),
      // Every turn gets a fresh JS session, so no damaged warm session survives a failure.
      markResidentDirty: () => {},
      consumeResidentDirty: () => false,
      invalidateResident: () => {},
      runWithResident: async (_id, fn) => {
        if (this.pending >= SHARED_REQUEST_LIMIT) throw new Error('Shared inference queue is full.');
        this.pending++;
        let loaded = false;
        try {
          for (;;) {
            signal.throwIfAborted();
            const outcome = await this.coordinator.withInference(async () => {
              if (this.resident?.key !== key) return undefined;
              const resident = this.resident;
              const run = async () => {
                signal.throwIfAborted();
                await resident.admission.flushPendingDisposals();
                if (resident.admission.pendingDisposalCount > 0)
                  throw new Error('Could not release a previous inference session.');
                const session = new ChatSession(resident.model);
                try {
                  // Flat speculative models need a full reset between independent owners.
                  if (resident.model.hasBlockPagedCache?.() !== true) await session.reset();
                  return { value: await fn(session, !loaded) };
                } finally {
                  await resident.admission.disposeSession(session);
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
                    admission: new SessionRegistry({
                      model,
                      maxQueueDepth: 16,
                      maxConcurrentDispatches:
                        Number.isSafeInteger(capacity) && capacity! > 1 ? Math.min(capacity!, 4) : 1,
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
    await resident.admission.flushPendingDisposals();
    if (resident.admission.pendingDisposalCount > 0) throw new Error('Could not release the resident model sessions.');
    await resident.host.dispose();
    this.resident = undefined;
    await resident.overlays.cleanup();
  }
}
