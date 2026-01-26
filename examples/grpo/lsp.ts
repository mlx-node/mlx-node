/**
 * Fast Octokit API Documentation Service using oxc-parser.
 *
 * This module provides a way to introspect the Octokit REST API types to get
 * parameter information, return types, and JSDoc descriptions for any method.
 *
 * Performance targets:
 * - Initialization: < 1 second
 * - Queries: < 10ms
 *
 * Uses oxc-parser (Rust-based) for dramatically faster parsing compared to
 * TypeScript's language service.
 */

import { parse } from 'oxc-parser';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

// Get __dirname equivalent for ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================================================
// Type Definitions
// ============================================================================

export interface LspToolArgs {
  /** Method path, e.g., "octokit.rest.pulls.list" */
  method: string;
  /** What to query: parameters or return type */
  kind: 'parameters' | 'return';
}

export interface ParameterInfo {
  name: string;
  type: string;
  required: boolean;
  description?: string;
}

export interface LspParametersResult {
  method: string;
  httpMethod: string;
  httpPath: string;
  description?: string;
  parameters: ParameterInfo[];
}

export interface LspReturnResult {
  method: string;
  returnType: string;
  description?: string;
}

export interface LspErrorResult {
  error: string;
  suggestion?: string;
  availableNamespaces?: string[];
  availableMethods?: string[];
}

export type LspResult = LspParametersResult | LspReturnResult | LspErrorResult;

interface ParsedMethod {
  namespace: string;
  method: string;
  description?: string;
  httpMethod: string;
  httpPath: string;
}

export interface LspService {
  methods: Map<string, ParsedMethod>;
  namespaces: Set<string>;
  methodsByNamespace: Map<string, string[]>;
  openApiContent: string;
  cache: Map<string, LspResult>;
}

// ============================================================================
// Path Resolution
// ============================================================================

/**
 * Find the node_modules directory by searching upward from the current file.
 */
function findNodeModules(): string {
  let dir = __dirname;
  while (dir !== path.dirname(dir)) {
    const nodeModules = path.join(dir, 'node_modules');
    if (fs.existsSync(nodeModules)) {
      return nodeModules;
    }
    dir = path.dirname(dir);
  }
  throw new Error('Could not find node_modules directory');
}

/**
 * Resolve a path within a package.
 */
function resolvePackagePath(packageName: string, internalPath: string): string {
  const nodeModules = findNodeModules();
  return path.join(nodeModules, packageName, internalPath);
}

// ============================================================================
// AST Node Types (from oxc-parser)
// ============================================================================

interface Comment {
  type: 'Block' | 'Line';
  value: string;
  start: number;
  end: number;
}

interface BaseNode {
  type: string;
  start: number;
  end: number;
}

interface Identifier extends BaseNode {
  type: 'Identifier';
  name: string;
}

interface Literal extends BaseNode {
  type: 'Literal';
  value: string | number | boolean | null;
}

interface TSLiteralType extends BaseNode {
  type: 'TSLiteralType';
  literal: Literal;
}

interface TSTypeReference extends BaseNode {
  type: 'TSTypeReference';
  typeName: Identifier;
}

interface TSIndexedAccessType extends BaseNode {
  type: 'TSIndexedAccessType';
  objectType: TSTypeReference | TSIndexedAccessType;
  indexType: TSLiteralType;
}

interface TSIntersectionType extends BaseNode {
  type: 'TSIntersectionType';
  types: Array<TSTypeReference | TSIndexedAccessType>;
}

interface TSTypeAnnotation extends BaseNode {
  type: 'TSTypeAnnotation';
  typeAnnotation: TSTypeLiteral | TSIntersectionType | BaseNode;
}

interface TSPropertySignature extends BaseNode {
  type: 'TSPropertySignature';
  key: Identifier;
  typeAnnotation?: TSTypeAnnotation;
}

interface TSTypeLiteral extends BaseNode {
  type: 'TSTypeLiteral';
  members: TSPropertySignature[];
}

interface TSTypeAliasDeclaration extends BaseNode {
  type: 'TSTypeAliasDeclaration';
  id: Identifier;
  typeAnnotation: TSTypeLiteral;
}

interface ExportNamedDeclaration extends BaseNode {
  type: 'ExportNamedDeclaration';
  declaration?: TSTypeAliasDeclaration;
}

interface Program extends BaseNode {
  type: 'Program';
  body: Array<TSTypeAliasDeclaration | ExportNamedDeclaration | BaseNode>;
}

// ============================================================================
// Initialization
// ============================================================================

/**
 * Initialize the LSP service by loading and parsing Octokit type files.
 */
export async function initLspService(): Promise<LspService> {
  const methodTypesPath = resolvePackagePath(
    '@octokit/plugin-rest-endpoint-methods',
    'dist-types/generated/method-types.d.ts',
  );
  const paramTypesPath = resolvePackagePath(
    '@octokit/plugin-rest-endpoint-methods',
    'dist-types/generated/parameters-and-response-types.d.ts',
  );
  const openApiTypesPath = resolvePackagePath('@octokit/openapi-types', 'types.d.ts');

  // Read files
  const methodTypesContent = fs.readFileSync(methodTypesPath, 'utf-8');
  const paramTypesContent = fs.readFileSync(paramTypesPath, 'utf-8');
  const openApiContent = fs.readFileSync(openApiTypesPath, 'utf-8');

  // Parse with oxc-parser
  const methodTypesResult = await parse('method-types.d.ts', methodTypesContent, {
    sourceType: 'module',
    lang: 'dts',
  });
  const paramTypesResult = await parse('param-types.d.ts', paramTypesContent, {
    sourceType: 'module',
    lang: 'dts',
  });

  // Build method index
  const { methods, namespaces, methodsByNamespace } = buildMethodIndex(
    methodTypesResult.program as Program,
    methodTypesResult.comments as Comment[],
    methodTypesContent,
    paramTypesResult.program as Program,
  );

  return {
    methods,
    namespaces,
    methodsByNamespace,
    openApiContent,
    cache: new Map(),
  };
}

// ============================================================================
// AST Parsing
// ============================================================================

/**
 * Build the method index from parsed AST.
 */
function buildMethodIndex(
  methodProgram: Program,
  comments: Comment[],
  sourceText: string,
  paramProgram: Program,
): {
  methods: Map<string, ParsedMethod>;
  namespaces: Set<string>;
  methodsByNamespace: Map<string, string[]>;
} {
  const methods = new Map<string, ParsedMethod>();
  const namespaces = new Set<string>();
  const methodsByNamespace = new Map<string, string[]>();

  // Build comment index: map node start position to preceding JSDoc comment
  const commentIndex = buildCommentIndex(comments, sourceText);

  // Build HTTP info index from parameters file
  const httpInfoIndex = buildHttpInfoIndex(paramProgram);

  // Find RestEndpointMethods type alias (may be wrapped in ExportNamedDeclaration)
  for (const node of methodProgram.body) {
    let decl: TSTypeAliasDeclaration | null = null;

    if (node.type === 'TSTypeAliasDeclaration') {
      decl = node as TSTypeAliasDeclaration;
    } else if (node.type === 'ExportNamedDeclaration') {
      const exportDecl = node as ExportNamedDeclaration;
      if (exportDecl.declaration?.type === 'TSTypeAliasDeclaration') {
        decl = exportDecl.declaration;
      }
    }

    if (decl && decl.id.name === 'RestEndpointMethods') {
      const typeLiteral = decl.typeAnnotation;
      if (typeLiteral.type === 'TSTypeLiteral') {
        // Iterate namespaces
        for (const nsMember of typeLiteral.members) {
          if (nsMember.type === 'TSPropertySignature' && nsMember.key.type === 'Identifier') {
            const namespace = nsMember.key.name;
            namespaces.add(namespace);
            methodsByNamespace.set(namespace, []);

            // Get namespace type literal
            const nsType = nsMember.typeAnnotation?.typeAnnotation;
            if (nsType && nsType.type === 'TSTypeLiteral') {
              const nsTypeLiteral = nsType as TSTypeLiteral;
              // Iterate methods
              for (const methodMember of nsTypeLiteral.members) {
                if (methodMember.type === 'TSPropertySignature' && methodMember.key.type === 'Identifier') {
                  const methodName = methodMember.key.name;
                  const key = `${namespace}.${methodName}`;

                  methodsByNamespace.get(namespace)!.push(methodName);

                  // Get JSDoc comment
                  const description = commentIndex.get(methodMember.start);

                  // Get HTTP info
                  const httpInfo = httpInfoIndex.get(key) || inferHttpInfo(namespace, methodName);

                  methods.set(key, {
                    namespace,
                    method: methodName,
                    description,
                    httpMethod: httpInfo.method,
                    httpPath: httpInfo.path,
                  });
                }
              }
            }
          }
        }
      }
    }
  }

  return { methods, namespaces, methodsByNamespace };
}

/**
 * Build an index of JSDoc comments by node start position.
 * Maps a node's start position to the JSDoc comment text that precedes it.
 */
function buildCommentIndex(comments: Comment[], sourceText: string): Map<number, string> {
  const index = new Map<number, string>();

  for (const comment of comments) {
    // Only process block comments (JSDoc style)
    if (comment.type !== 'Block') continue;

    // Check if it's a JSDoc comment (starts with *)
    if (!comment.value.startsWith('*')) continue;

    // Find the next non-whitespace position after the comment
    let pos = comment.end;
    while (pos < sourceText.length && /\s/.test(sourceText[pos])) {
      pos++;
    }

    // Extract and clean the JSDoc content
    const cleanedComment = comment.value
      .replace(/^\*\s*/, '')
      .replace(/\s*$/m, '')
      .replace(/^\s*\*\s?/gm, '')
      .trim();

    index.set(pos, cleanedComment);
  }

  return index;
}

/**
 * Build an index of HTTP method/path info from the parameters file.
 */
function buildHttpInfoIndex(paramProgram: Program): Map<string, { method: string; path: string }> {
  const index = new Map<string, { method: string; path: string }>();

  // Find RestEndpointMethodTypes type alias (may be wrapped in ExportNamedDeclaration)
  for (const node of paramProgram.body) {
    let decl: TSTypeAliasDeclaration | null = null;

    if (node.type === 'TSTypeAliasDeclaration') {
      decl = node as TSTypeAliasDeclaration;
    } else if (node.type === 'ExportNamedDeclaration') {
      const exportDecl = node as ExportNamedDeclaration;
      if (exportDecl.declaration?.type === 'TSTypeAliasDeclaration') {
        decl = exportDecl.declaration;
      }
    }

    if (decl && decl.id.name === 'RestEndpointMethodTypes') {
      const typeLiteral = decl.typeAnnotation;
      if (typeLiteral.type === 'TSTypeLiteral') {
        // Iterate namespaces
        for (const nsMember of typeLiteral.members) {
          if (nsMember.type === 'TSPropertySignature' && nsMember.key.type === 'Identifier') {
            const namespace = nsMember.key.name;

            // Get namespace type literal
            const nsType = nsMember.typeAnnotation?.typeAnnotation;
            if (nsType && nsType.type === 'TSTypeLiteral') {
              const nsTypeLiteral = nsType as TSTypeLiteral;
              // Iterate methods
              for (const methodMember of nsTypeLiteral.members) {
                if (methodMember.type === 'TSPropertySignature' && methodMember.key.type === 'Identifier') {
                  const methodName = methodMember.key.name;
                  const key = `${namespace}.${methodName}`;

                  // Extract HTTP info from the parameters property
                  const httpInfo = extractHttpInfoFromMember(methodMember);
                  if (httpInfo) {
                    index.set(key, httpInfo);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return index;
}

/**
 * Extract HTTP method and path from a method member's parameters property.
 * Looks for patterns like: Endpoints["GET /repos/{owner}/{repo}/pulls"]["parameters"]
 */
function extractHttpInfoFromMember(methodMember: TSPropertySignature): { method: string; path: string } | null {
  const methodType = methodMember.typeAnnotation?.typeAnnotation;
  if (!methodType || methodType.type !== 'TSTypeLiteral') return null;

  const methodTypeLiteral = methodType as TSTypeLiteral;
  for (const prop of methodTypeLiteral.members) {
    if (prop.type === 'TSPropertySignature' && prop.key.type === 'Identifier') {
      if (prop.key.name === 'parameters') {
        return extractEndpointString(prop);
      }
    }
  }
  return null;
}

/**
 * Extract the endpoint string from a parameters property type annotation.
 */
function extractEndpointString(prop: TSPropertySignature): { method: string; path: string } | null {
  const typeAnnotation = prop.typeAnnotation?.typeAnnotation;
  if (!typeAnnotation) return null;

  // Handle intersection type: RequestParameters & Endpoints[...][...]
  if (typeAnnotation.type === 'TSIntersectionType') {
    const intersection = typeAnnotation as TSIntersectionType;
    for (const t of intersection.types) {
      if (t.type === 'TSIndexedAccessType') {
        const result = extractFromIndexedAccess(t as TSIndexedAccessType);
        if (result) return result;
      }
    }
  }

  // Handle direct indexed access
  if (typeAnnotation.type === 'TSIndexedAccessType') {
    return extractFromIndexedAccess(typeAnnotation as TSIndexedAccessType);
  }

  return null;
}

/**
 * Extract HTTP method and path from TSIndexedAccessType.
 * Handles: Endpoints["GET /path"]["parameters"]
 */
function extractFromIndexedAccess(node: TSIndexedAccessType): { method: string; path: string } | null {
  // The outer indexed access has indexType = "parameters" or "response"
  // The inner (objectType) has indexType = "GET /path" string
  const objectType = node.objectType;
  if (objectType.type === 'TSIndexedAccessType') {
    const innerIndexed = objectType as TSIndexedAccessType;
    const indexType = innerIndexed.indexType;
    if (indexType.type === 'TSLiteralType') {
      const literal = indexType.literal;
      if (literal.type === 'Literal' && typeof literal.value === 'string') {
        // Parse "GET /repos/{owner}/{repo}/pulls"
        const match = literal.value.match(/^(GET|POST|PUT|PATCH|DELETE)\s+(.+)$/);
        if (match) {
          return { method: match[1], path: match[2] };
        }
      }
    }
  }
  return null;
}

/**
 * Infer HTTP method and path from method name.
 */
function inferHttpInfo(namespace: string, methodName: string): { method: string; path: string } {
  let method = 'GET';
  if (methodName.startsWith('create') || methodName.startsWith('add')) method = 'POST';
  else if (methodName.startsWith('update') || methodName.startsWith('set')) method = 'PATCH';
  else if (methodName.startsWith('delete') || methodName.startsWith('remove')) method = 'DELETE';
  else if (methodName.startsWith('merge') || methodName.startsWith('enable')) method = 'PUT';

  return { method, path: `/repos/{owner}/{repo}/${namespace}` };
}

// ============================================================================
// Query Functions
// ============================================================================

/**
 * Query a method for parameters or return type information.
 */
export function queryMethod(service: LspService, args: LspToolArgs): LspResult {
  const cacheKey = `${args.method}:${args.kind}`;
  if (service.cache.has(cacheKey)) {
    return service.cache.get(cacheKey)!;
  }

  const result = queryMethodImpl(service, args);
  service.cache.set(cacheKey, result);
  return result;
}

function queryMethodImpl(service: LspService, args: LspToolArgs): LspResult {
  // Parse method path: octokit.rest.pulls.list -> ['pulls', 'list']
  const parts = args.method.replace(/^octokit\.rest\./, '').split('.');
  if (parts.length !== 2) {
    return {
      error: `Invalid method format: "${args.method}". Expected "octokit.rest.<namespace>.<method>"`,
      suggestion: 'Use format like "octokit.rest.pulls.list"',
    };
  }

  const [namespace, methodName] = parts;

  // Check namespace exists
  if (!service.namespaces.has(namespace)) {
    const suggestion = findClosestMatch(namespace, Array.from(service.namespaces));
    return {
      error: `Namespace "${namespace}" not found`,
      suggestion: suggestion ? `Did you mean "${suggestion}"?` : undefined,
      availableNamespaces: Array.from(service.namespaces).slice(0, 20),
    };
  }

  // Check method exists
  const methodKey = `${namespace}.${methodName}`;
  const methodInfo = service.methods.get(methodKey);
  if (!methodInfo) {
    const availableMethods = service.methodsByNamespace.get(namespace) || [];
    const suggestion = findClosestMatch(methodName, availableMethods);
    return {
      error: `Method "${methodName}" not found in "${namespace}"`,
      suggestion: suggestion ? `Did you mean "${suggestion}"?` : undefined,
      availableMethods: availableMethods.slice(0, 30),
    };
  }

  if (args.kind === 'parameters') {
    return extractParameters(service, namespace, methodName, methodInfo);
  } else {
    return extractReturnType(namespace, methodName, methodInfo);
  }
}

// ============================================================================
// Parameter Extraction (Regex-based for performance)
// ============================================================================

function extractParameters(
  service: LspService,
  namespace: string,
  methodName: string,
  methodInfo: ParsedMethod,
): LspParametersResult {
  // Use regex-based parsing of the openapi-types file (fast)
  const parameters = parseParametersFromOpenApiRegex(service.openApiContent, namespace, methodName);

  return {
    method: `octokit.rest.${namespace}.${methodName}`,
    httpMethod: methodInfo.httpMethod,
    httpPath: methodInfo.httpPath,
    description: methodInfo.description?.split('\n')[0],
    parameters,
  };
}

/**
 * Parse parameters from openapi-types using regex (fast approach).
 */
function parseParametersFromOpenApiRegex(content: string, namespace: string, methodName: string): ParameterInfo[] {
  const parameters: ParameterInfo[] = [];

  // Look for the operation DEFINITION (not reference)
  const operationDefPattern = `"${namespace}/${methodName}": {`;
  const opIndex = content.indexOf(operationDefPattern);
  if (opIndex === -1) return parameters;

  // Extract a chunk of text after the operation name
  const chunk = content.substring(opIndex, opIndex + 3000);

  // Parse path parameters
  const pathMatch = chunk.match(/path:\s*\{([\s\S]*?)\n\s*\};/);
  if (pathMatch) {
    parseParamsFromBlock(pathMatch[1], parameters, true);
  }

  // Parse query parameters
  const queryMatch = chunk.match(/query\??:\s*\{([\s\S]*?)\n\s*\};/);
  if (queryMatch) {
    parseParamsFromBlock(queryMatch[1], parameters, false);
  }

  // Parse request body parameters
  const bodyMatch = chunk.match(/"application\/json":\s*\{([\s\S]*?)\n\s*\};/);
  if (bodyMatch) {
    parseParamsFromBlock(bodyMatch[1], parameters, false);
  }

  // Sort: required first, then alphabetical
  parameters.sort((a, b) => {
    if (a.required !== b.required) return a.required ? -1 : 1;
    return a.name.localeCompare(b.name);
  });

  return parameters;
}

/**
 * Parse individual parameters from a block of text.
 */
function parseParamsFromBlock(block: string, parameters: ParameterInfo[], allRequired: boolean): void {
  const lines = block.split(';');

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // Parse JSDoc description
    const descMatch = trimmed.match(/@description\s+([^*]+)/);
    const description = descMatch ? descMatch[1].trim() : undefined;

    // Extract the actual property definition
    const propMatch = trimmed.match(/(\w+)(\??)\s*:\s*(.+)$/);
    if (propMatch) {
      const name = propMatch[1];
      const optional = propMatch[2] === '?';
      let type = propMatch[3].trim();

      // Clean up component references
      type = type.replace(/components\["parameters"\]\["([^"]+)"\]/g, (_, ref) => {
        const paramTypes: Record<string, string> = {
          owner: 'string',
          repo: 'string',
          'per-page': 'number',
          page: 'number',
          since: 'string',
          'pull-number': 'number',
          'issue-number': 'number',
          direction: '"asc" | "desc"',
          sort: '"created" | "updated"',
        };
        return paramTypes[ref] || ref;
      });

      parameters.push({
        name,
        type,
        required: allRequired && !optional,
        description,
      });
    }
  }
}

// ============================================================================
// Return Type Extraction
// ============================================================================

function extractReturnType(namespace: string, methodName: string, methodInfo: ParsedMethod): LspReturnResult {
  const returnType = `Promise<OctokitResponse<${namespace}/${methodName}>>`;

  return {
    method: `octokit.rest.${namespace}.${methodName}`,
    returnType,
    description: methodInfo.description?.split('\n')[0],
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Compute Levenshtein distance for typo suggestions.
 */
function levenshteinDistance(a: string, b: string): number {
  const matrix: number[][] = [];

  for (let i = 0; i <= b.length; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= a.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b[i - 1] === a[j - 1]) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(matrix[i - 1][j - 1] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j] + 1);
      }
    }
  }

  return matrix[b.length][a.length];
}

function findClosestMatch(target: string, candidates: string[]): string | undefined {
  let bestMatch: string | undefined;
  let bestDistance = Infinity;

  const maxDistance = Math.max(2, Math.floor(target.length / 2));

  for (const candidate of candidates) {
    const distance = levenshteinDistance(target.toLowerCase(), candidate.toLowerCase());
    if (distance < bestDistance && distance <= maxDistance) {
      bestDistance = distance;
      bestMatch = candidate;
    }
  }

  return bestMatch;
}

// ============================================================================
// Tool Call Interface
// ============================================================================

/**
 * Execute a tool call and return a JSON string response.
 */
export function executeToolCall(service: LspService, args: LspToolArgs): string {
  const result = queryMethod(service, args);
  return JSON.stringify(result, null, 2);
}
