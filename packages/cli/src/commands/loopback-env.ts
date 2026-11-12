const LOOPBACK_HOSTS = ['127.0.0.1', 'localhost', '::1'] as const;
const PROXY_VARIABLES = new Set(['http_proxy', 'https_proxy', 'all_proxy']);

/**
 * A child environment that reaches the local server without a proxy (Splash
 * `_reach_loopback_directly`): where the environment names a proxy
 * (http_proxy, https_proxy or all_proxy, in either case), NO_PROXY and
 * no_proxy each gain the loopback hosts they lack, so no proxy receives the
 * server's key. Each reads itself, else the other; one that exempts every
 * host as `*` stays as it is. An environment naming no proxy is returned
 * unchanged: the macOS system proxy's own exceptions then decide.
 */
export function withLoopbackNoProxy(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  let namesProxy = false;
  for (const [name, value] of Object.entries(env)) {
    if (value && PROXY_VARIABLES.has(name.toLowerCase())) {
      namesProxy = true;
      break;
    }
  }
  if (!namesProxy) return env;
  const found = {
    NO_PROXY: env.NO_PROXY || env.no_proxy || '',
    no_proxy: env.no_proxy || env.NO_PROXY || '',
  };
  const out: NodeJS.ProcessEnv = { ...env };
  for (const [name, value] of Object.entries(found)) {
    const hosts: string[] = [];
    for (const part of value.split(',')) {
      const host = part.trim();
      if (host) hosts.push(host);
    }
    const missing = LOOPBACK_HOSTS.filter((host) => !hosts.includes(host));
    if (missing.length > 0 && !(hosts.length === 1 && hosts[0] === '*')) {
      out[name] = [...hosts, ...missing].join(',');
    }
  }
  return out;
}
