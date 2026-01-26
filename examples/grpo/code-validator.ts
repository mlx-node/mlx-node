/**
 * Code Validator for GitHub Tool Training
 *
 * Full semantic validation of generated ESM JavaScript code using oxc-parser.
 * Validates:
 * - Syntax correctness
 * - Import validation (only 'octokit', '@napi-rs/simple-git', './utils' allowed)
 * - Required utils import: import { owner, repo, currentBranch, octokit } from './utils'
 * - Default export function structure
 * - Async/await usage
 * - Error handling patterns (typed errors)
 */

import { parseSync, Visitor } from 'oxc-parser';

// Allowed import sources
const ALLOWED_IMPORTS = new Set(['./utils']);

const REQUIRED_IMPORTS = new Set(['owner', 'repo', 'currentBranch', 'octokit']);

export interface ValidationResult {
  /** Overall validity */
  valid: boolean;
  /** Contains required utils import */
  hasUtilsImport: boolean;
  /** Contains required imports */
  hasRequiredImports: boolean;
  /** Syntax is correct (no parse errors) */
  syntaxValid: boolean;
  /** Only allowed imports used */
  importsValid: boolean;
  /** Has default export */
  hasDefaultExport: boolean;
  /** Default export is an async function */
  hasAsyncFunction: boolean;
  /** Function returns an object */
  returnsObject: boolean;
  /** Uses async/await */
  hasAsyncAwait: boolean;
  /** Has try/catch with typed error handling */
  hasErrorHandling: boolean;
  /** Has typed error checks (404, 403, etc.) */
  hasTypedErrorChecks: boolean;
  /** List of validation errors */
  errors: string[];
  /** List of detected imports */
  imports: string[];
}

interface ValidationState {
  hasDefaultExport: boolean;
  defaultExportIsAsync: boolean;
  defaultExportHasReturn: boolean;
  hasRequiredImports: boolean;
  imports: string[];
  invalidImports: string[];
  hasTryCatch: boolean;
  hasTypedErrorChecks: boolean;
  hasAwait: boolean;
  returnStatements: number;
}

/**
 * Validate generated JavaScript code for the GitHub tool training.
 */
export function validateGeneratedCode(code: string): ValidationResult {
  const errors: string[] = [];
  const state: ValidationState = {
    hasDefaultExport: false,
    defaultExportIsAsync: false,
    defaultExportHasReturn: false,
    hasRequiredImports: false,
    imports: [],
    invalidImports: [],
    hasTryCatch: false,
    hasTypedErrorChecks: false,
    hasAwait: false,
    returnStatements: 0,
  };

  // Step 1: Parse the code
  let parseResult;
  try {
    parseResult = parseSync('generated.js', code, {
      sourceType: 'module',
      showSemanticErrors: true,
    });
  } catch (error) {
    return {
      valid: false,
      hasUtilsImport: false,
      hasRequiredImports: false,
      syntaxValid: false,
      importsValid: false,
      hasDefaultExport: false,
      hasAsyncFunction: false,
      returnsObject: false,
      hasAsyncAwait: false,
      hasErrorHandling: false,
      hasTypedErrorChecks: false,
      errors: [`Parse error: ${error instanceof Error ? error.message : String(error)}`],
      imports: [],
    };
  }

  // Check for parse errors
  const syntaxValid = parseResult.errors.length === 0;
  if (!syntaxValid) {
    for (const err of parseResult.errors) {
      errors.push(`Syntax error: ${err.message}`);
    }
  }

  // Step 2: Walk the AST to collect information
  const visitor = new Visitor({
    // Track imports
    ImportDeclaration(node) {
      const source = node.source.value;
      state.imports.push(source);
      if (!ALLOWED_IMPORTS.has(source)) {
        state.invalidImports.push(source);
      }
      if (
        node.specifiers.length === 4 &&
        node.specifiers.every((specifier) => REQUIRED_IMPORTS.has(specifier.local.name))
      ) {
        state.hasRequiredImports = true;
      }
    },

    // Track default export
    ExportDefaultDeclaration(node) {
      state.hasDefaultExport = true;

      // Check if it's an async function
      const decl = node.declaration;
      if (decl.type === 'FunctionDeclaration' || decl.type === 'FunctionExpression') {
        state.defaultExportIsAsync = decl.async === true;
      } else if (decl.type === 'ArrowFunctionExpression') {
        state.defaultExportIsAsync = decl.async === true;
      }
    },

    // Track try-catch blocks
    TryStatement(_node) {
      state.hasTryCatch = true;
    },

    // Track await expressions
    AwaitExpression(_node) {
      state.hasAwait = true;
    },

    // Track return statements
    ReturnStatement(_node) {
      state.returnStatements++;
      state.defaultExportHasReturn = true;
    },

    // Track typed error checks (error.status === 404, etc.)
    BinaryExpression(node) {
      // Check for patterns like: error.status === 404
      if (node.operator === '===' || node.operator === '==') {
        const left = node.left;
        const right = node.right;

        // Check if left side is error.status or *.status
        if (
          left.type === 'MemberExpression' &&
          left.property.type === 'Identifier' &&
          left.property.name === 'status'
        ) {
          // Check if right side is a common HTTP status code (Literal type in ESTree)
          if (right.type === 'Literal' && typeof right.value === 'number') {
            const statusCodes = [400, 401, 403, 404, 422, 500, 502, 503];
            if (statusCodes.includes(right.value)) {
              state.hasTypedErrorChecks = true;
            }
          }
        }
      }
    },
  });

  try {
    visitor.visit(parseResult.program);
  } catch (error) {
    errors.push(`AST walk error: ${error instanceof Error ? error.message : String(error)}`);
  }

  // Step 3: Build validation result
  const importsValid = state.invalidImports.length === 0;
  if (!importsValid) {
    for (const imp of state.invalidImports) {
      errors.push(`Invalid import: '${imp}' (only 'octokit', '@napi-rs/simple-git', and './utils' are allowed)`);
    }
  }

  // Check for utils import (just verify './utils' is imported)
  const hasUtilsImport = state.imports.includes('./utils');
  if (!hasUtilsImport) {
    errors.push(`Missing utils import: import { ... } from './utils'`);
  }

  if (!state.hasDefaultExport) {
    errors.push('Missing default export');
  }

  if (state.hasDefaultExport && !state.defaultExportIsAsync) {
    errors.push('Default export must be an async function');
  }

  if (!state.hasTryCatch) {
    errors.push('Missing try-catch block for error handling');
  }

  if (state.hasTryCatch && !state.hasTypedErrorChecks) {
    errors.push('Missing typed error checks (e.g., error.status === 404)');
  }

  if (!state.hasAwait) {
    errors.push('No await expressions found (code should use async operations)');
  }

  if (state.returnStatements === 0) {
    errors.push('No return statement found');
  }

  const valid = syntaxValid && importsValid && hasUtilsImport && state.hasDefaultExport && state.defaultExportIsAsync;

  return {
    valid,
    hasUtilsImport,
    syntaxValid,
    importsValid,
    hasRequiredImports: state.hasRequiredImports,
    hasDefaultExport: state.hasDefaultExport,
    hasAsyncFunction: state.defaultExportIsAsync,
    returnsObject: state.returnStatements > 0,
    hasAsyncAwait: state.hasAwait,
    hasErrorHandling: state.hasTryCatch,
    hasTypedErrorChecks: state.hasTypedErrorChecks,
    errors,
    imports: state.imports,
  };
}

// Example valid code for testing (uses utils import)
export const EXAMPLE_VALID_CODE = `import { owner, repo, currentBranch, octokit } from './utils';

export default async function() {
  try {
    // Find PR for current branch
    const { data: prs } = await octokit.rest.pulls.list({
      owner,
      repo,
      head: \`\${owner}:\${currentBranch}\`,
      state: 'open',
    });

    if (prs.length === 0) {
      return { error: 'no_pr', message: 'No open PR for this branch' };
    }

    const { data: comments } = await octokit.rest.pulls.listReviewComments({
      owner,
      repo,
      pull_number: prs[0].number,
    });

    return { comments, prNumber: prs[0].number };
  } catch (error) {
    if (error.status === 404) {
      return { error: 'not_found', message: 'PR not found' };
    }
    if (error.status === 403) {
      return { error: 'forbidden', message: 'Rate limited or no permission' };
    }
    throw error;
  }
}`;
