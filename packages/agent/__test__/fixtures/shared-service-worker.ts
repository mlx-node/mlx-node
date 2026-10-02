/** A real child process using the production election/transport, without model weights. */
import { appendFile } from 'node:fs/promises';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

import { startSharedService } from '../../src/provider/shared-service.js';

const directory = process.argv[2]!;
try {
  const service = await startSharedService({
    directory,
    port: Number(process.argv[3]),
    loadBackend: async () => {
      await appendFile(join(directory, 'loads'), `${process.pid}\n`);
      return {
        busy: () => false,
        close: async () => {},
        async *stream(request) {
          await delay(25);
          yield { error: `${process.pid}:${request.options.sessionId}` };
        },
      };
    },
  });
  process.stdout.write(`ready:${service.endpoint.pid}\n`);
  process.once('SIGTERM', () => { void service.close(); });
} catch (error) {
  if ((error as NodeJS.ErrnoException).code !== 'EADDRINUSE') throw error;
  process.stdout.write('contender\n');
}
