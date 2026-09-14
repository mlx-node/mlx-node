import type { ExtensionAPI, ExtensionContext, InlineExtension } from '@earendil-works/pi-coding-agent';

export interface DelegateCallerPermissions {
  source: 'codex';
  /** Opaque Codex profile/backend when supplied; otherwise just inherited process permissions. */
  profile: string;
  networkDisabled: boolean;
}

/** Child processes stay inside Codex's actual sandbox; this metadata grants no OS access. */
export function delegateCallerPermissions(env: NodeJS.ProcessEnv = process.env): DelegateCallerPermissions | undefined {
  if (!env.CODEX_THREAD_ID?.trim()) return undefined;
  const profile = env.CODEX_PERMISSION_PROFILE?.trim();
  // Permission metadata is optional in Codex launches. The thread identifies
  // the caller; subprocesses inherit its real sandbox regardless of these labels.
  const sandbox = env.CODEX_SANDBOX;
  return {
    source: 'codex',
    profile: profile || (sandbox === 'seatbelt' || sandbox === 'landlock' ? `sandbox:${sandbox}` : 'inherited'),
    networkDisabled: env.CODEX_SANDBOX_NETWORK_DISABLED === '1',
  };
}

const READ_TOOLS = new Set(['read', 'grep', 'find', 'ls']);

/** Only failed tool executions qualify; quoted CI logs in a successful result do not. */
export function delegationBlocker(text: string): boolean {
  return /permission denied|operation not permitted|\bEACCES\b|\bEPERM\b|network access was denied|sandbox.*(?:denied|blocked)|(?:HTTP\s+|status(?: code)?[: ]+)(?:401|403)\b|gh auth login|could not resolve host|could not resolve proxy|failed to connect|error connecting to api\.github\.com/i.test(
    text,
  );
}

export function createDelegationExtension(options: { callerApproved?: boolean } = {}): InlineExtension {
  // Capture before any model/tool code runs. Never consult model-written settings.
  const caller = delegateCallerPermissions();
  const explicitlyApproved = options.callerApproved === true || process.env.MLX_AGENT_AUTO_APPROVE === '1';
  return {
    name: 'mlx-delegation',
    factory: (pi: ExtensionAPI) => {
      let handoff: string | undefined;

      const stop = (reason: string, ctx: ExtensionContext): string => {
        if (!handoff) {
          const sessionFile = ctx.sessionManager.getSessionFile();
          handoff = `Delegation incomplete: ${reason}\nContinue this task in the calling agent. Do not retry with changed permissions or another agent.${sessionFile ? `\nSaved evidence and tool results: ${sessionFile}` : ''}`;
          pi.appendEntry('mlx-delegate-handoff', { reason, caller, sessionFile });
          // Print mode otherwise shows only the final assistant message. A deterministic
          // diagnostic and failure status let the caller recover without another inference.
          process.stderr.write(`${handoff}\n`);
          process.exitCode = 1;
        }
        ctx.abort();
        return handoff;
      };

      pi.on('before_agent_start', (event) => {
        handoff = undefined;
        const permissions = caller
          ? `Tool execution inherits the calling Codex process's permissions (${caller.profile}).${caller.networkDisabled ? ' The caller disables network access.' : ''} Additional approval must be handled by the calling agent; this worker cannot request escalation.`
          : explicitlyApproved
            ? "The caller explicitly approved tool execution for this bounded task. Child processes inherit its OS sandbox, environment and credentials; this does not copy the calling agent's tool approval rules or grant additional access. Stay within the task authorization."
            : 'No inherited caller permission context is available. Tools requiring approval will stop this task and return a handoff.';
        return { systemPrompt: `${event.systemPrompt}\n\n${permissions}` };
      });

      pi.on('tool_call', (event, ctx) => {
        if (handoff) return { block: true, terminate: true, reason: handoff };
        if (event.toolName === 'subagent') {
          return { block: true, terminate: true, reason: stop('This worker cannot create subagents.', ctx) };
        }
        if (READ_TOOLS.has(event.toolName)) return undefined;
        if (!caller && !explicitlyApproved) {
          return {
            block: true,
            terminate: true,
            reason: stop(
              `No caller permission context is available to authorize ${event.toolName}. The calling agent can use --caller-approved after approving this bounded task.`,
              ctx,
            ),
          };
        }
        if (event.toolName === 'bash') {
          const input = event.input as { command?: unknown };
          if (typeof input.command === 'string') {
            // Preserve failures through pipelines such as `gh ... | head`, and
            // stop a multi-command script before later output masks a denial.
            // This changes shell options only; cwd, environment and sandbox remain inherited.
            input.command = `set -e -o pipefail\n${input.command}`;
          }
        }
        return undefined;
      });

      pi.on('tool_result', (event, ctx) => {
        if (!event.isError) return undefined;
        const text = event.content
          .filter((part) => part.type === 'text')
          .map((part) => part.text)
          .join('\n');
        if (!delegationBlocker(text)) return undefined;
        const reason = stop(`${event.toolName} was blocked. ${text.slice(0, 2000)}`, ctx);
        return { content: [{ type: 'text', text: reason }], isError: true };
      });
    },
  };
}
