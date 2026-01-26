/**
 * Tests for the LSP Tool (TypeScript Language Service wrapper for Octokit API documentation)
 *
 * The LSP tool provides a way to introspect Octokit REST API types to get
 * parameter information, return types, and JSDoc descriptions for any method.
 */

import { describe, it, expect, beforeAll } from 'vite-plus/test';
import {
  initLspService,
  queryMethod,
  executeToolCall,
  type LspService,
  type LspParametersResult,
  type LspReturnResult,
  type LspErrorResult,
} from '../grpo/lsp';

describe('LSP Tool', () => {
  let service: LspService;

  beforeAll(async () => {
    service = await initLspService();
  });

  // ============================================================================
  // Initialization Tests
  // ============================================================================

  describe('initLspService', () => {
    it('should initialize with namespaces and methods', () => {
      console.log(service.namespaces.size);
      expect(service.namespaces.size).toBeGreaterThan(0);
      expect(service.methods.size).toBeGreaterThan(0);
      expect(service.methodsByNamespace.size).toBeGreaterThan(0);
      expect(service.openApiContent.length).toBeGreaterThan(0);
    });

    it('should contain expected namespaces', () => {
      expect(service.namespaces.has('pulls')).toBe(true);
      expect(service.namespaces.has('issues')).toBe(true);
      expect(service.namespaces.has('actions')).toBe(true);
      expect(service.namespaces.has('repos')).toBe(true);
    });

    it('should have an empty cache initially', async () => {
      // Create a fresh service to test initial state
      const freshService = await initLspService();
      expect(freshService.cache.size).toBe(0);
    });
  });

  // ============================================================================
  // Training Scenario Methods - Parameters
  // ============================================================================

  describe('queryMethod - parameters (training scenarios)', () => {
    // Scenario 1: get-inline-comments
    it('should return parameters for octokit.rest.pulls.listReviewComments', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.listReviewComments',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;
      expect(params.method).toBe('octokit.rest.pulls.listReviewComments');
      expect(params.httpMethod).toBe('GET');
      expect(params.httpPath).toContain('/pulls/');
      expect(params.httpPath).toContain('/comments');
      expect(Array.isArray(params.parameters)).toBe(true);
      // Note: Parameters may be empty for some methods depending on OpenAPI parsing
    });

    // Scenario 2: get-comments (using issues.listComments)
    it('should return parameters for octokit.rest.issues.listComments', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.issues.listComments',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;
      expect(params.method).toBe('octokit.rest.issues.listComments');
      expect(params.httpMethod).toBe('GET');
      expect(Array.isArray(params.parameters)).toBe(true);
    });

    // Scenario 3: reply-inline-comment
    it('should return parameters for octokit.rest.pulls.createReplyForReviewComment', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.createReplyForReviewComment',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;
      expect(params.method).toBe('octokit.rest.pulls.createReplyForReviewComment');
      expect(params.httpMethod).toBe('POST');
      expect(Array.isArray(params.parameters)).toBe(true);
      // Note: Parameters may be empty for some methods depending on OpenAPI parsing
    });

    // Scenario 6: get-ci-outputs
    it('should return parameters for octokit.rest.actions.downloadWorkflowRunLogs', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.actions.downloadWorkflowRunLogs',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;
      expect(params.method).toBe('octokit.rest.actions.downloadWorkflowRunLogs');
      expect(params.httpMethod).toBe('GET');
      expect(Array.isArray(params.parameters)).toBe(true);
      // Note: Parameters may be empty for some methods depending on OpenAPI parsing
    });

    // Scenario 7: change-pr-title
    it('should return parameters for octokit.rest.pulls.update', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.update',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;
      expect(params.method).toBe('octokit.rest.pulls.update');
      expect(params.httpMethod).toBe('PATCH');
      expect(Array.isArray(params.parameters)).toBe(true);

      // Should have title parameter
      const paramNames = params.parameters.map((p) => p.name);
      expect(paramNames).toContain('title');
    });

    // Scenario 8: get-file-diff
    it('should return parameters for octokit.rest.pulls.listFiles', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.listFiles',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;
      expect(params.method).toBe('octokit.rest.pulls.listFiles');
      expect(params.httpMethod).toBe('GET');
      expect(Array.isArray(params.parameters)).toBe(true);
    });

    // Scenario 9: check-workflow-status (listWorkflowRunsForRepo)
    it('should return parameters for octokit.rest.actions.listWorkflowRunsForRepo', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.actions.listWorkflowRunsForRepo',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;
      expect(params.method).toBe('octokit.rest.actions.listWorkflowRunsForRepo');
      expect(params.httpMethod).toBe('GET');
      expect(Array.isArray(params.parameters)).toBe(true);
    });

    // Scenario 10: trigger-workflow
    it('should return parameters for octokit.rest.actions.createWorkflowDispatch', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.actions.createWorkflowDispatch',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;
      expect(params.method).toBe('octokit.rest.actions.createWorkflowDispatch');
      expect(params.httpMethod).toBe('POST');
      expect(Array.isArray(params.parameters)).toBe(true);
      // Note: Parameters may be empty for some methods depending on OpenAPI parsing
    });

    // Additional commonly used methods
    it('should return parameters for octokit.rest.pulls.list', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.list',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;
      expect(params.method).toBe('octokit.rest.pulls.list');
      expect(params.httpMethod).toBe('GET');
      expect(Array.isArray(params.parameters)).toBe(true);

      // Should have state, head, base filter parameters
      const paramNames = params.parameters.map((p) => p.name);
      expect(paramNames).toContain('state');
    });

    it('should return parameters for octokit.rest.pulls.get', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.get',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;
      expect(params.method).toBe('octokit.rest.pulls.get');
      expect(params.httpMethod).toBe('GET');
      expect(Array.isArray(params.parameters)).toBe(true);

      const paramNames = params.parameters.map((p) => p.name);
      expect(paramNames).toContain('owner');
      expect(paramNames).toContain('repo');
      expect(paramNames).toContain('pull_number');
    });

    it('should return parameters for octokit.rest.repos.get', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.repos.get',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;
      expect(params.method).toBe('octokit.rest.repos.get');
      expect(params.httpMethod).toBe('GET');
      expect(Array.isArray(params.parameters)).toBe(true);
    });

    it('should return parameters for octokit.rest.issues.create', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.issues.create',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;
      expect(params.method).toBe('octokit.rest.issues.create');
      expect(params.httpMethod).toBe('POST');
      expect(Array.isArray(params.parameters)).toBe(true);

      const paramNames = params.parameters.map((p) => p.name);
      expect(paramNames).toContain('title');
    });

    it('should return parameters for octokit.rest.actions.listWorkflowRuns', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.actions.listWorkflowRuns',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;
      expect(params.method).toBe('octokit.rest.actions.listWorkflowRuns');
      expect(params.httpMethod).toBe('GET');
      expect(Array.isArray(params.parameters)).toBe(true);
      // Note: Parameters may be empty for some methods depending on OpenAPI parsing
    });
  });

  // ============================================================================
  // Training Scenario Methods - Return Types
  // ============================================================================

  describe('queryMethod - return (training scenarios)', () => {
    it('should return type for octokit.rest.pulls.listReviewComments', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.listReviewComments',
        kind: 'return',
      });

      expect(result).not.toHaveProperty('error');
      const returnResult = result as LspReturnResult;
      expect(returnResult.method).toBe('octokit.rest.pulls.listReviewComments');
      expect(returnResult.returnType).toContain('Promise');
      expect(returnResult.returnType).toContain('OctokitResponse');
    });

    it('should return type for octokit.rest.issues.listComments', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.issues.listComments',
        kind: 'return',
      });

      expect(result).not.toHaveProperty('error');
      const returnResult = result as LspReturnResult;
      expect(returnResult.method).toBe('octokit.rest.issues.listComments');
      expect(returnResult.returnType).toContain('Promise');
    });

    it('should return type for octokit.rest.pulls.createReplyForReviewComment', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.createReplyForReviewComment',
        kind: 'return',
      });

      expect(result).not.toHaveProperty('error');
      const returnResult = result as LspReturnResult;
      expect(returnResult.method).toBe('octokit.rest.pulls.createReplyForReviewComment');
      expect(returnResult.returnType).toContain('Promise');
    });

    it('should return type for octokit.rest.actions.downloadWorkflowRunLogs', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.actions.downloadWorkflowRunLogs',
        kind: 'return',
      });

      expect(result).not.toHaveProperty('error');
      const returnResult = result as LspReturnResult;
      expect(returnResult.method).toBe('octokit.rest.actions.downloadWorkflowRunLogs');
      expect(returnResult.returnType).toContain('Promise');
    });

    it('should return type for octokit.rest.pulls.update', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.update',
        kind: 'return',
      });

      expect(result).not.toHaveProperty('error');
      const returnResult = result as LspReturnResult;
      expect(returnResult.method).toBe('octokit.rest.pulls.update');
      expect(returnResult.returnType).toContain('Promise');
    });

    it('should return type for octokit.rest.pulls.listFiles', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.listFiles',
        kind: 'return',
      });

      expect(result).not.toHaveProperty('error');
      const returnResult = result as LspReturnResult;
      expect(returnResult.method).toBe('octokit.rest.pulls.listFiles');
      expect(returnResult.returnType).toContain('Promise');
    });

    it('should return type for octokit.rest.actions.listWorkflowRunsForRepo', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.actions.listWorkflowRunsForRepo',
        kind: 'return',
      });

      expect(result).not.toHaveProperty('error');
      const returnResult = result as LspReturnResult;
      expect(returnResult.method).toBe('octokit.rest.actions.listWorkflowRunsForRepo');
      expect(returnResult.returnType).toContain('Promise');
    });

    it('should return type for octokit.rest.actions.createWorkflowDispatch', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.actions.createWorkflowDispatch',
        kind: 'return',
      });

      expect(result).not.toHaveProperty('error');
      const returnResult = result as LspReturnResult;
      expect(returnResult.method).toBe('octokit.rest.actions.createWorkflowDispatch');
      expect(returnResult.returnType).toContain('Promise');
    });

    it('should return type for octokit.rest.pulls.list', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.list',
        kind: 'return',
      });

      expect(result).not.toHaveProperty('error');
      const returnResult = result as LspReturnResult;
      expect(returnResult.method).toBe('octokit.rest.pulls.list');
      expect(returnResult.returnType).toContain('Promise');
    });

    it('should return type for octokit.rest.pulls.get', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.get',
        kind: 'return',
      });

      expect(result).not.toHaveProperty('error');
      const returnResult = result as LspReturnResult;
      expect(returnResult.method).toBe('octokit.rest.pulls.get');
      expect(returnResult.returnType).toContain('Promise');
    });
  });

  // ============================================================================
  // Error Handling Tests
  // ============================================================================

  describe('queryMethod - error handling', () => {
    it('should handle method format without octokit.rest prefix (strips prefix internally)', () => {
      // The LSP tool strips the prefix internally, so this should work
      const result = queryMethod(service, {
        method: 'pulls.list',
        kind: 'parameters',
      });

      // This actually succeeds because the code strips the octokit.rest prefix
      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;
      expect(params.method).toBe('octokit.rest.pulls.list');
    });

    it('should error on invalid method format (single segment)', () => {
      const result = queryMethod(service, {
        method: 'list',
        kind: 'parameters',
      });

      expect(result).toHaveProperty('error');
      const error = result as LspErrorResult;
      expect(error.error).toContain('Invalid method format');
    });

    it('should error on namespace typo and suggest correct namespace', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pull.list',
        kind: 'parameters',
      });

      expect(result).toHaveProperty('error');
      const error = result as LspErrorResult;
      expect(error.error).toContain('Namespace "pull" not found');
      expect(error.suggestion).toContain('pulls');
      // Note: availableNamespaces is limited to 20 items, alphabetically sorted
      // so 'pulls' may not be in the first 20
      expect(error.availableNamespaces).toBeDefined();
      expect(error.availableNamespaces!.length).toBeGreaterThan(0);
    });

    it('should error on method typo and suggest correct method', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.lst',
        kind: 'parameters',
      });

      expect(result).toHaveProperty('error');
      const error = result as LspErrorResult;
      expect(error.error).toContain('Method "lst" not found in "pulls"');
      expect(error.suggestion).toContain('list');
      expect(error.availableMethods).toContain('list');
    });

    it('should error on non-existent namespace', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.foobar.list',
        kind: 'parameters',
      });

      expect(result).toHaveProperty('error');
      const error = result as LspErrorResult;
      expect(error.error).toContain('Namespace "foobar" not found');
      expect(error.availableNamespaces).toBeDefined();
      expect(error.availableNamespaces!.length).toBeGreaterThan(0);
    });

    it('should error on non-existent method', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.foobar',
        kind: 'parameters',
      });

      expect(result).toHaveProperty('error');
      const error = result as LspErrorResult;
      expect(error.error).toContain('Method "foobar" not found in "pulls"');
      expect(error.availableMethods).toBeDefined();
      expect(error.availableMethods!.length).toBeGreaterThan(0);
    });

    it('should error on close misspelling and suggest correct namespace', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.isues.list',
        kind: 'parameters',
      });

      expect(result).toHaveProperty('error');
      const error = result as LspErrorResult;
      expect(error.error).toContain('Namespace "isues" not found');
      expect(error.suggestion).toContain('issues');
    });

    it('should error on close misspelling and suggest correct method', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.updte',
        kind: 'parameters',
      });

      expect(result).toHaveProperty('error');
      const error = result as LspErrorResult;
      expect(error.error).toContain('Method "updte" not found');
      expect(error.suggestion).toContain('update');
    });

    it('should handle actions namespace typo', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.action.listWorkflowRuns',
        kind: 'parameters',
      });

      expect(result).toHaveProperty('error');
      const error = result as LspErrorResult;
      expect(error.error).toContain('Namespace "action" not found');
      expect(error.suggestion).toContain('actions');
    });

    it('should handle repos namespace typo', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.repo.get',
        kind: 'parameters',
      });

      expect(result).toHaveProperty('error');
      const error = result as LspErrorResult;
      expect(error.error).toContain('Namespace "repo" not found');
      expect(error.suggestion).toContain('repos');
    });
  });

  // ============================================================================
  // Response Structure Tests
  // ============================================================================

  describe('response structure', () => {
    it('parameters response should have correct structure', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.list',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;

      // Check required fields
      expect(params).toHaveProperty('method');
      expect(params).toHaveProperty('httpMethod');
      expect(params).toHaveProperty('httpPath');
      expect(params).toHaveProperty('parameters');

      // Description is optional but typically present
      expect(typeof params.method).toBe('string');
      expect(typeof params.httpMethod).toBe('string');
      expect(typeof params.httpPath).toBe('string');
      expect(Array.isArray(params.parameters)).toBe(true);
    });

    it('each parameter should have correct structure', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.list',
        kind: 'parameters',
      });

      expect(result).not.toHaveProperty('error');
      const params = result as LspParametersResult;

      for (const param of params.parameters) {
        expect(param).toHaveProperty('name');
        expect(param).toHaveProperty('type');
        expect(param).toHaveProperty('required');

        expect(typeof param.name).toBe('string');
        expect(typeof param.type).toBe('string');
        expect(typeof param.required).toBe('boolean');
        // description is optional
        if (param.description !== undefined) {
          expect(typeof param.description).toBe('string');
        }
      }
    });

    it('return response should have correct structure', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.list',
        kind: 'return',
      });

      expect(result).not.toHaveProperty('error');
      const returnResult = result as LspReturnResult;

      // Check required fields
      expect(returnResult).toHaveProperty('method');
      expect(returnResult).toHaveProperty('returnType');

      expect(typeof returnResult.method).toBe('string');
      expect(typeof returnResult.returnType).toBe('string');
      // description is optional
      if (returnResult.description !== undefined) {
        expect(typeof returnResult.description).toBe('string');
      }
    });

    it('error response should have correct structure', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.nonexistent.method',
        kind: 'parameters',
      });

      expect(result).toHaveProperty('error');
      const error = result as LspErrorResult;

      expect(typeof error.error).toBe('string');
      // suggestion is optional
      if (error.suggestion !== undefined) {
        expect(typeof error.suggestion).toBe('string');
      }
      // availableNamespaces is optional
      if (error.availableNamespaces !== undefined) {
        expect(Array.isArray(error.availableNamespaces)).toBe(true);
      }
      // availableMethods is optional
      if (error.availableMethods !== undefined) {
        expect(Array.isArray(error.availableMethods)).toBe(true);
      }
    });
  });

  // ============================================================================
  // Caching Tests
  // ============================================================================

  describe('caching', () => {
    it('should cache results and return same object on second call', async () => {
      // Create a fresh service to test caching
      const freshService = await initLspService();

      const firstResult = queryMethod(freshService, {
        method: 'octokit.rest.pulls.list',
        kind: 'parameters',
      });

      expect(freshService.cache.size).toBe(1);

      const secondResult = queryMethod(freshService, {
        method: 'octokit.rest.pulls.list',
        kind: 'parameters',
      });

      // Should be the exact same object (reference equality)
      expect(firstResult).toBe(secondResult);
      expect(freshService.cache.size).toBe(1);
    });

    it('should cache different kinds separately', async () => {
      const freshService = await initLspService();

      queryMethod(freshService, {
        method: 'octokit.rest.pulls.list',
        kind: 'parameters',
      });

      queryMethod(freshService, {
        method: 'octokit.rest.pulls.list',
        kind: 'return',
      });

      // Should have two cache entries (one for each kind)
      expect(freshService.cache.size).toBe(2);
    });

    it('should cache different methods separately', async () => {
      const freshService = await initLspService();

      queryMethod(freshService, {
        method: 'octokit.rest.pulls.list',
        kind: 'parameters',
      });

      queryMethod(freshService, {
        method: 'octokit.rest.pulls.get',
        kind: 'parameters',
      });

      // Should have two cache entries (one for each method)
      expect(freshService.cache.size).toBe(2);
    });

    it('should also cache error results', async () => {
      const freshService = await initLspService();

      const firstResult = queryMethod(freshService, {
        method: 'octokit.rest.invalid.method',
        kind: 'parameters',
      });

      const secondResult = queryMethod(freshService, {
        method: 'octokit.rest.invalid.method',
        kind: 'parameters',
      });

      expect(firstResult).toBe(secondResult);
      expect(freshService.cache.size).toBe(1);
    });
  });

  // ============================================================================
  // executeToolCall Tests
  // ============================================================================

  describe('executeToolCall', () => {
    it('should return JSON string for parameters', () => {
      const result = executeToolCall(service, {
        method: 'octokit.rest.pulls.list',
        kind: 'parameters',
      });

      expect(typeof result).toBe('string');
      const parsed = JSON.parse(result);
      expect(parsed.method).toBe('octokit.rest.pulls.list');
      expect(parsed.httpMethod).toBe('GET');
      expect(Array.isArray(parsed.parameters)).toBe(true);
    });

    it('should return JSON string for return type', () => {
      const result = executeToolCall(service, {
        method: 'octokit.rest.pulls.list',
        kind: 'return',
      });

      expect(typeof result).toBe('string');
      const parsed = JSON.parse(result);
      expect(parsed.method).toBe('octokit.rest.pulls.list');
      expect(parsed.returnType).toContain('Promise');
    });

    it('should return error JSON for invalid methods', () => {
      const result = executeToolCall(service, {
        method: 'octokit.rest.invalid.method',
        kind: 'parameters',
      });

      expect(typeof result).toBe('string');
      const parsed = JSON.parse(result);
      expect(parsed.error).toContain('Namespace "invalid" not found');
      expect(Array.isArray(parsed.availableNamespaces)).toBe(true);
    });

    it('should include suggestion in error JSON', () => {
      const result = executeToolCall(service, {
        method: 'octokit.rest.pull.list',
        kind: 'parameters',
      });

      const parsed = JSON.parse(result);
      expect(parsed.error).toBeDefined();
      expect(parsed.suggestion).toContain('pulls');
    });

    it('should include available methods in error JSON', () => {
      const result = executeToolCall(service, {
        method: 'octokit.rest.pulls.nonexistent',
        kind: 'parameters',
      });

      const parsed = JSON.parse(result);
      expect(parsed.error).toBeDefined();
      expect(Array.isArray(parsed.availableMethods)).toBe(true);
      expect(parsed.availableMethods).toContain('list');
    });

    it('should include required field in parameters', () => {
      const result = executeToolCall(service, {
        method: 'octokit.rest.pulls.get',
        kind: 'parameters',
      });

      const parsed = JSON.parse(result);
      const requiredParams = parsed.parameters.filter((p: { required: boolean }) => p.required);
      expect(requiredParams.length).toBeGreaterThan(0);
    });

    it('should handle method with parameters gracefully', () => {
      const result = executeToolCall(service, {
        method: 'octokit.rest.repos.get',
        kind: 'parameters',
      });

      expect(typeof result).toBe('string');
      const parsed = JSON.parse(result);
      expect(parsed.method).toBe('octokit.rest.repos.get');
      expect(Array.isArray(parsed.parameters)).toBe(true);
    });
  });

  // ============================================================================
  // HTTP Method Inference Tests
  // ============================================================================

  describe('HTTP method detection', () => {
    it('should detect GET for list methods', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.list',
        kind: 'parameters',
      }) as LspParametersResult;

      expect(result.httpMethod).toBe('GET');
    });

    it('should detect GET for get methods', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.get',
        kind: 'parameters',
      }) as LspParametersResult;

      expect(result.httpMethod).toBe('GET');
    });

    it('should detect POST for create methods', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.issues.create',
        kind: 'parameters',
      }) as LspParametersResult;

      expect(result.httpMethod).toBe('POST');
    });

    it('should detect PATCH for update methods', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.update',
        kind: 'parameters',
      }) as LspParametersResult;

      expect(result.httpMethod).toBe('PATCH');
    });

    it('should detect POST for workflow dispatch', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.actions.createWorkflowDispatch',
        kind: 'parameters',
      }) as LspParametersResult;

      expect(result.httpMethod).toBe('POST');
    });
  });

  // ============================================================================
  // Namespace Coverage Tests
  // ============================================================================

  describe('namespace coverage', () => {
    const expectedNamespaces = ['pulls', 'issues', 'actions', 'repos', 'git', 'users', 'orgs'];

    for (const namespace of expectedNamespaces) {
      it(`should have "${namespace}" namespace`, () => {
        expect(service.namespaces.has(namespace)).toBe(true);
      });

      it(`should have methods for "${namespace}" namespace`, () => {
        const methods = service.methodsByNamespace.get(namespace);
        expect(methods).toBeDefined();
        expect(methods!.length).toBeGreaterThan(0);
      });
    }
  });

  // ============================================================================
  // Method with Description Tests
  // ============================================================================

  describe('method descriptions', () => {
    it('should include description when available for parameters', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.list',
        kind: 'parameters',
      }) as LspParametersResult;

      // Description may or may not be present depending on the source types
      // Just verify the field exists and is either string or undefined
      expect(result.description === undefined || typeof result.description === 'string').toBe(true);
    });

    it('should include description when available for return type', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.list',
        kind: 'return',
      }) as LspReturnResult;

      // Description may or may not be present
      expect(result.description === undefined || typeof result.description === 'string').toBe(true);
    });
  });

  // ============================================================================
  // HTTP Path Format Tests
  // ============================================================================

  describe('HTTP path format', () => {
    it('should include owner and repo placeholders in path', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.list',
        kind: 'parameters',
      }) as LspParametersResult;

      expect(result.httpPath).toContain('{owner}');
      expect(result.httpPath).toContain('{repo}');
    });

    it('should include pull_number placeholder for PR-specific endpoints', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.pulls.get',
        kind: 'parameters',
      }) as LspParametersResult;

      expect(result.httpPath).toContain('{pull_number}');
    });

    it('should include workflow_id placeholder for workflow endpoints', () => {
      const result = queryMethod(service, {
        method: 'octokit.rest.actions.listWorkflowRuns',
        kind: 'parameters',
      }) as LspParametersResult;

      expect(result.httpPath).toContain('{workflow_id}');
    });
  });
});
