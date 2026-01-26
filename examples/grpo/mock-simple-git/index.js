/**
 * Mock @napi-rs/simple-git for Execution-Based Reward Validation
 *
 * This mock provides fake git repository context without needing
 * an actual git repository. Values come from MOCK_CONTEXT env var.
 *
 * Environment variables:
 * - MOCK_CONTEXT: JSON string with { owner, repo, branch }
 */

// Load mock context from environment
let context = { owner: 'test-owner', repo: 'test-repo', branch: 'test-feature' };

try {
  if (process.env.MOCK_CONTEXT) {
    context = JSON.parse(process.env.MOCK_CONTEXT);
  }
} catch (e) {
  console.error('[mock-simple-git] Failed to parse MOCK_CONTEXT:', e.message);
}

/**
 * Mock Reference class (returned by repo.head())
 */
class MockReference {
  constructor(branchName) {
    this._branchName = branchName;
  }

  shorthand() {
    return this._branchName;
  }

  name() {
    return `refs/heads/${this._branchName}`;
  }
}

/**
 * Mock Remote class (returned by repo.remotes() or repo.findRemote())
 */
class MockRemote {
  constructor(remoteName, remoteUrl) {
    this._name = remoteName;
    this._url = remoteUrl;
  }

  name() {
    return this._name;
  }

  url() {
    return this._url;
  }
}

/**
 * Mock Repository class
 *
 * Provides fake implementations that return values from MOCK_CONTEXT
 */
export class Repository {
  constructor(path) {
    // Don't actually open a repository - just store the path
    this._path = path;
  }

  /**
   * Get remote URL (direct method)
   * Returns a fake GitHub URL using the mock context
   */
  remoteUrl() {
    // Return a GitHub URL that will parse to the mock owner/repo
    return `git@github.com:${context.owner}/${context.repo}.git`;
  }

  /**
   * Get all remotes
   * Returns an array of MockRemote objects
   */
  remotes() {
    const url = `git@github.com:${context.owner}/${context.repo}.git`;
    return [new MockRemote('origin', url)];
  }

  /**
   * Find a remote by name
   * Returns a MockRemote or null
   */
  findRemote(remoteName) {
    if (remoteName === 'origin') {
      const url = `git@github.com:${context.owner}/${context.repo}.git`;
      return new MockRemote('origin', url);
    }
    return null;
  }

  /**
   * Get HEAD reference
   * Returns a mock reference with the mock branch name
   */
  head() {
    return new MockReference(context.branch);
  }

  /**
   * Get current branch name (alternative method)
   */
  currentBranch() {
    return context.branch;
  }

  /**
   * Check if path is inside a work tree
   */
  isPathInWorkTree() {
    return true;
  }

  /**
   * Get the work tree path
   */
  workTree() {
    return this._path;
  }
}

export default { Repository };
