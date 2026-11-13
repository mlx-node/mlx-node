/**
 * The inference endpoint an agent process was started with. The host keeps only a fingerprint (pid, url and a hash of
 * the token) to notice a restarted or replaced sidecar; the token itself goes only into the agent's environment.
 */

import { createHash } from 'node:crypto';

/** The shape of `~/.mlx-node/desktop/inference.json` (`@mlx-node/server/host/desktop-endpoint`). */
export interface InferenceEndpoint {
  version: 1;
  pid: number;
  url: string;
  token: string;
  models: Array<{ name: string; path: string; base: string; variant: string; format: 'gguf' | 'mlx' }>;
}

/** Reads the endpoint file; null when it is missing or fails any check. Never throws. */
export type EndpointReader = () => Promise<InferenceEndpoint | null>;

export interface EndpointFingerprint {
  pid: number;
  url: string;
  tokenSha256: string;
}

export function endpointFingerprint(endpoint: InferenceEndpoint): EndpointFingerprint {
  return {
    pid: endpoint.pid,
    url: endpoint.url,
    tokenSha256: createHash('sha256').update(endpoint.token).digest('hex'),
  };
}

export function sameEndpoint(a: EndpointFingerprint, b: EndpointFingerprint): boolean {
  return a.pid === b.pid && a.url === b.url && a.tokenSha256 === b.tokenSha256;
}

/** A reader that never throws, around any reader. */
export function safeReader(read: EndpointReader): EndpointReader {
  return async () => {
    try {
      return await read();
    } catch {
      return null;
    }
  };
}
