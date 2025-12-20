/**
 * Expanded ast-grep pattern dataset for GRPO training
 *
 * Contains 50+ unique patterns organized by difficulty:
 * - Basic: 15 patterns (function decl, const/let, imports, class)
 * - Intermediate: 20 patterns (destructuring, spread, async/await, promises)
 * - Advanced: 15 patterns (React hooks, HOFs, decorators, generics)
 */

export interface AstGrepPattern {
  id: string;
  category: 'basic' | 'intermediate' | 'advanced';
  language: 'javascript' | 'typescript';
  description: string;
  pattern: string;
  codeContext: string;
  expectedMatches: number;
}

// ============================================================================
// Real Code Contexts (realistic multi-line examples)
// ============================================================================

export const CODE_CONTEXTS = {
  reactComponent: `import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { fetchUser } from './api';

interface UserProps {
  userId: string;
  onLoad?: () => void;
}

export function UserProfile({ userId, onLoad }: UserProps) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const formattedName = useMemo(() => {
    return user ? user.name.toUpperCase() : '';
  }, [user]);

  const handleRefresh = useCallback(() => {
    setLoading(true);
    fetchUser(userId).then(setUser);
  }, [userId]);

  useEffect(() => {
    let mounted = true;

    async function loadUser() {
      try {
        const data = await fetchUser(userId);
        if (mounted) {
          setUser(data);
          setLoading(false);
          onLoad?.();
        }
      } catch (err) {
        if (mounted) {
          setError(err);
          setLoading(false);
        }
      }
    }

    loadUser();
    return () => { mounted = false; };
  }, [userId, onLoad]);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  return <div>{formattedName}</div>;
}`,

  expressServer: `import express from 'express';
import cors from 'cors';
import { router as userRouter } from './routes/user';
import { errorHandler } from './middleware/error';

const app = express();

app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.post('/api/users', async (req, res) => {
  const { name, email } = req.body;
  try {
    const user = await createUser({ name, email });
    res.status(201).json(user);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.use('/api/users', userRouter);
app.use(errorHandler);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(\`Server running on port \${PORT}\`);
});`,

  typescriptInterfaces: `interface User {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
}

interface Post {
  id: string;
  title: string;
  content: string;
  author: User;
}

type UserWithPosts = User & {
  posts: Post[];
};

type CreateUserInput = Omit<User, 'id' | 'createdAt'>;

interface Repository<T> {
  find(id: string): Promise<T | null>;
  findAll(): Promise<T[]>;
  create(data: Partial<T>): Promise<T>;
  update(id: string, data: Partial<T>): Promise<T>;
  delete(id: string): Promise<void>;
}

class UserRepository implements Repository<User> {
  private users: Map<string, User> = new Map();

  async find(id: string): Promise<User | null> {
    return this.users.get(id) || null;
  }

  async findAll(): Promise<User[]> {
    return Array.from(this.users.values());
  }

  async create(data: Partial<User>): Promise<User> {
    const user = { ...data, id: crypto.randomUUID(), createdAt: new Date() } as User;
    this.users.set(user.id, user);
    return user;
  }

  async update(id: string, data: Partial<User>): Promise<User> {
    const user = this.users.get(id);
    if (!user) throw new Error('User not found');
    const updated = { ...user, ...data };
    this.users.set(id, updated);
    return updated;
  }

  async delete(id: string): Promise<void> {
    this.users.delete(id);
  }
}`,

  errorHandling: `async function fetchData(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(\`HTTP error: \${response.status}\`);
    }
    return await response.json();
  } catch (error) {
    console.error('Fetch failed:', error);
    throw error;
  }
}

function processWithRetry(fn, maxRetries = 3) {
  let attempts = 0;

  return async function(...args) {
    while (attempts < maxRetries) {
      try {
        return await fn(...args);
      } catch (error) {
        attempts++;
        if (attempts === maxRetries) {
          throw new Error(\`Failed after \${maxRetries} attempts: \${error.message}\`);
        }
        await new Promise(r => setTimeout(r, 1000 * attempts));
      }
    }
  };
}

class ValidationError extends Error {
  constructor(message, field) {
    super(message);
    this.name = 'ValidationError';
    this.field = field;
  }
}

function validate(data) {
  if (!data.name) {
    throw new ValidationError('Name is required', 'name');
  }
  if (!data.email) {
    throw new ValidationError('Email is required', 'email');
  }
  return true;
}`,

  modernJavaScript: `// Destructuring
const { name, age, ...rest } = user;
const [first, second, ...remaining] = items;

// Spread operator
const merged = { ...defaults, ...config };
const combined = [...array1, ...array2];

// Arrow functions
const double = x => x * 2;
const add = (a, b) => a + b;
const greet = name => \`Hello, \${name}!\`;

// Template literals
const message = \`User \${name} is \${age} years old\`;

// Optional chaining
const city = user?.address?.city;
const result = obj?.method?.();

// Nullish coalescing
const value = input ?? defaultValue;
const port = process.env.PORT ?? 3000;

// Array methods
const doubled = numbers.map(n => n * 2);
const evens = numbers.filter(n => n % 2 === 0);
const sum = numbers.reduce((acc, n) => acc + n, 0);
const hasNegative = numbers.some(n => n < 0);
const allPositive = numbers.every(n => n > 0);
const found = items.find(item => item.id === targetId);

// Promises
const fetchUser = id => fetch(\`/api/users/\${id}\`).then(r => r.json());
Promise.all([fetchUser(1), fetchUser(2)]).then(users => console.log(users));
Promise.race([timeout(5000), fetchData()]).then(result => console.log(result));`,

  classPatterns: `class Animal {
  constructor(name) {
    this.name = name;
  }

  speak() {
    console.log(\`\${this.name} makes a sound\`);
  }
}

class Dog extends Animal {
  constructor(name, breed) {
    super(name);
    this.breed = breed;
  }

  speak() {
    console.log(\`\${this.name} barks\`);
  }

  fetch() {
    return \`\${this.name} fetches the ball\`;
  }
}

class Cat extends Animal {
  speak() {
    console.log(\`\${this.name} meows\`);
  }
}

class Singleton {
  static instance = null;

  static getInstance() {
    if (!Singleton.instance) {
      Singleton.instance = new Singleton();
    }
    return Singleton.instance;
  }
}`,
};

// ============================================================================
// Basic Patterns (15)
// ============================================================================

const basicPatterns: AstGrepPattern[] = [
  {
    id: 'basic-function-decl',
    category: 'basic',
    language: 'javascript',
    description: 'Find all function declarations',
    pattern: 'function $NAME($$$ARGS) { $$$BODY }',
    codeContext: CODE_CONTEXTS.expressServer,
    expectedMatches: 0, // Express uses arrow functions
  },
  {
    id: 'basic-async-function',
    category: 'basic',
    language: 'javascript',
    description: 'Find async function declarations',
    pattern: 'async function $NAME($$$ARGS) { $$$BODY }',
    codeContext: CODE_CONTEXTS.errorHandling,
    expectedMatches: 1, // fetchData
  },
  {
    id: 'basic-const-decl',
    category: 'basic',
    language: 'javascript',
    description: 'Find const declarations',
    pattern: 'const $NAME = $VALUE',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 17,
  },
  {
    id: 'basic-let-decl',
    category: 'basic',
    language: 'javascript',
    description: 'Find let declarations',
    pattern: 'let $NAME = $VALUE',
    codeContext: CODE_CONTEXTS.errorHandling,
    expectedMatches: 1, // let attempts
  },
  {
    id: 'basic-arrow-simple',
    category: 'basic',
    language: 'javascript',
    description: 'Find simple arrow functions (single expression)',
    pattern: '$PARAM => $BODY',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 11,
  },
  {
    id: 'basic-import-named',
    category: 'basic',
    language: 'javascript',
    description: 'Find named imports',
    pattern: 'import { $$$NAMES } from $SOURCE',
    codeContext: CODE_CONTEXTS.reactComponent,
    expectedMatches: 2,
  },
  {
    id: 'basic-import-default',
    category: 'basic',
    language: 'javascript',
    description: 'Find default imports',
    pattern: 'import $NAME from $SOURCE',
    codeContext: CODE_CONTEXTS.expressServer,
    expectedMatches: 2, // express, cors
  },
  {
    id: 'basic-export-function',
    category: 'basic',
    language: 'javascript',
    description: 'Find exported function declarations',
    pattern: 'export function $NAME($$$ARGS) { $$$BODY }',
    codeContext: CODE_CONTEXTS.reactComponent,
    expectedMatches: 1, // UserProfile
  },
  {
    id: 'basic-class-decl',
    category: 'basic',
    language: 'javascript',
    description: 'Find class declarations',
    pattern: 'class $NAME { $$$BODY }',
    codeContext: CODE_CONTEXTS.classPatterns,
    expectedMatches: 2, // Animal, Singleton
  },
  {
    id: 'basic-class-extends',
    category: 'basic',
    language: 'javascript',
    description: 'Find classes that extend another class',
    pattern: 'class $NAME extends $PARENT { $$$BODY }',
    codeContext: CODE_CONTEXTS.classPatterns,
    expectedMatches: 2, // Dog, Cat
  },
  {
    id: 'basic-if-statement',
    category: 'basic',
    language: 'javascript',
    description: 'Find if statements',
    pattern: 'if ($COND) { $$$BODY }',
    codeContext: CODE_CONTEXTS.errorHandling,
    expectedMatches: 6,
  },
  {
    id: 'basic-return',
    category: 'basic',
    language: 'javascript',
    description: 'Find return statements',
    pattern: 'return $VALUE',
    codeContext: CODE_CONTEXTS.errorHandling,
    expectedMatches: 5,
  },
  {
    id: 'basic-throw',
    category: 'basic',
    language: 'javascript',
    description: 'Find throw statements',
    pattern: 'throw $ERROR',
    codeContext: CODE_CONTEXTS.errorHandling,
    expectedMatches: 4,
  },
  {
    id: 'basic-console-log',
    category: 'basic',
    language: 'javascript',
    description: 'Find console.log calls',
    pattern: 'console.log($$$ARGS)',
    codeContext: CODE_CONTEXTS.classPatterns,
    expectedMatches: 3,
  },
  {
    id: 'basic-template-literal',
    category: 'basic',
    language: 'javascript',
    description: 'Find template literals with expressions',
    pattern: '`$$$CONTENT`',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 4,
  },
];

// ============================================================================
// Intermediate Patterns (20)
// ============================================================================

const intermediatePatterns: AstGrepPattern[] = [
  {
    id: 'inter-destructure-object',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find object destructuring assignments',
    pattern: 'const { $$$PROPS } = $OBJ',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 1,
  },
  {
    id: 'inter-destructure-array',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find array destructuring assignments',
    pattern: 'const [$$$ITEMS] = $ARR',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 1,
  },
  {
    id: 'inter-spread-object',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find object spread usage',
    pattern: '{ ...$OBJ }',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 2,
  },
  {
    id: 'inter-spread-array',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find array spread usage',
    pattern: '[...$ARR]',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 1, // Only standalone spreads
  },
  {
    id: 'inter-async-await',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find await expressions',
    pattern: 'await $EXPR',
    codeContext: CODE_CONTEXTS.errorHandling,
    expectedMatches: 4,
  },
  {
    id: 'inter-try-catch',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find try-catch blocks',
    pattern: 'try { $$$TRY } catch ($ERR) { $$$CATCH }',
    codeContext: CODE_CONTEXTS.errorHandling,
    expectedMatches: 2,
  },
  {
    id: 'inter-promise-then',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find promise .then() chains',
    pattern: '$PROMISE.then($$$ARGS)',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 3,
  },
  {
    id: 'inter-optional-chain',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find optional chaining',
    pattern: '$OBJ?.$PROP',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 3,
  },
  {
    id: 'inter-nullish-coalesce',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find nullish coalescing',
    pattern: '$LEFT ?? $RIGHT',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 2,
  },
  {
    id: 'inter-array-map',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find array .map() calls',
    pattern: '$ARR.map($$$ARGS)',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 1,
  },
  {
    id: 'inter-array-filter',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find array .filter() calls',
    pattern: '$ARR.filter($$$ARGS)',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 1,
  },
  {
    id: 'inter-array-reduce',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find array .reduce() calls',
    pattern: '$ARR.reduce($$$ARGS)',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 1,
  },
  {
    id: 'inter-array-find',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find array .find() calls',
    pattern: '$ARR.find($$$ARGS)',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 1,
  },
  {
    id: 'inter-method-def',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find method definitions in classes',
    pattern: '$NAME($$$ARGS) { $$$BODY }',
    codeContext: CODE_CONTEXTS.classPatterns,
    expectedMatches: 8,
  },
  {
    id: 'inter-constructor',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find constructor methods',
    pattern: 'constructor($$$ARGS) { $$$BODY }',
    codeContext: CODE_CONTEXTS.classPatterns,
    expectedMatches: 2,
  },
  {
    id: 'inter-super-call',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find super() calls',
    pattern: 'super($$$ARGS)',
    codeContext: CODE_CONTEXTS.classPatterns,
    expectedMatches: 1,
  },
  {
    id: 'inter-new-error',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find new Error() instantiations',
    pattern: 'new Error($$$ARGS)',
    codeContext: CODE_CONTEXTS.errorHandling,
    expectedMatches: 2,
  },
  {
    id: 'inter-static-method',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find static methods',
    pattern: 'static $NAME($$$ARGS) { $$$BODY }',
    codeContext: CODE_CONTEXTS.classPatterns,
    expectedMatches: 1, // getInstance
  },
  {
    id: 'inter-static-property',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find static properties',
    pattern: 'static $NAME = $VALUE',
    codeContext: CODE_CONTEXTS.classPatterns,
    expectedMatches: 1, // instance = null
  },
  {
    id: 'inter-express-route',
    category: 'intermediate',
    language: 'javascript',
    description: 'Find Express route handlers',
    pattern: 'app.$METHOD($PATH, $$$HANDLERS)',
    codeContext: CODE_CONTEXTS.expressServer,
    expectedMatches: 4, // use, get, post, use, use
  },
];

// ============================================================================
// Advanced Patterns (15)
// ============================================================================

const advancedPatterns: AstGrepPattern[] = [
  {
    id: 'adv-use-state',
    category: 'advanced',
    language: 'javascript',
    description: 'Find React useState hooks',
    pattern: 'const [$STATE, $SETTER] = useState($INIT)',
    codeContext: CODE_CONTEXTS.reactComponent,
    expectedMatches: 3, // user, loading, error
  },
  {
    id: 'adv-use-effect',
    category: 'advanced',
    language: 'javascript',
    description: 'Find React useEffect hooks',
    pattern: 'useEffect($$$ARGS)',
    codeContext: CODE_CONTEXTS.reactComponent,
    expectedMatches: 1,
  },
  {
    id: 'adv-use-callback',
    category: 'advanced',
    language: 'javascript',
    description: 'Find React useCallback hooks',
    pattern: 'useCallback($$$ARGS)',
    codeContext: CODE_CONTEXTS.reactComponent,
    expectedMatches: 1,
  },
  {
    id: 'adv-use-memo',
    category: 'advanced',
    language: 'javascript',
    description: 'Find React useMemo hooks',
    pattern: 'useMemo($$$ARGS)',
    codeContext: CODE_CONTEXTS.reactComponent,
    expectedMatches: 1,
  },
  {
    id: 'adv-jsx-element',
    category: 'advanced',
    language: 'javascript',
    description: 'Find JSX div elements',
    pattern: '<div>$$$CHILDREN</div>',
    codeContext: CODE_CONTEXTS.reactComponent,
    expectedMatches: 3,
  },
  {
    id: 'adv-interface',
    category: 'advanced',
    language: 'typescript',
    description: 'Find TypeScript interfaces',
    pattern: 'interface $NAME { $$$BODY }',
    codeContext: CODE_CONTEXTS.typescriptInterfaces,
    expectedMatches: 4, // User, Post, Repository, UserWithPosts not matched (type alias)
  },
  {
    id: 'adv-type-alias',
    category: 'advanced',
    language: 'typescript',
    description: 'Find TypeScript type aliases',
    pattern: 'type $NAME = $TYPE',
    codeContext: CODE_CONTEXTS.typescriptInterfaces,
    expectedMatches: 2, // UserWithPosts, CreateUserInput
  },
  {
    id: 'adv-generic-interface',
    category: 'advanced',
    language: 'typescript',
    description: 'Find generic interface definitions',
    pattern: 'interface $NAME<$$$PARAMS> { $$$BODY }',
    codeContext: CODE_CONTEXTS.typescriptInterfaces,
    expectedMatches: 1, // Repository<T>
  },
  {
    id: 'adv-implements',
    category: 'advanced',
    language: 'typescript',
    description: 'Find classes implementing interfaces',
    pattern: 'class $NAME implements $INTERFACE { $$$BODY }',
    codeContext: CODE_CONTEXTS.typescriptInterfaces,
    expectedMatches: 1, // UserRepository
  },
  {
    id: 'adv-async-method',
    category: 'advanced',
    language: 'typescript',
    description: 'Find async class methods',
    pattern: 'async $NAME($$$ARGS): $RETURN { $$$BODY }',
    codeContext: CODE_CONTEXTS.typescriptInterfaces,
    expectedMatches: 5,
  },
  {
    id: 'adv-promise-all',
    category: 'advanced',
    language: 'javascript',
    description: 'Find Promise.all() calls',
    pattern: 'Promise.all($$$ARGS)',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 1,
  },
  {
    id: 'adv-promise-race',
    category: 'advanced',
    language: 'javascript',
    description: 'Find Promise.race() calls',
    pattern: 'Promise.race($$$ARGS)',
    codeContext: CODE_CONTEXTS.modernJavaScript,
    expectedMatches: 1,
  },
  {
    id: 'adv-custom-error',
    category: 'advanced',
    language: 'javascript',
    description: 'Find custom error class definitions',
    pattern: 'class $NAME extends Error { $$$BODY }',
    codeContext: CODE_CONTEXTS.errorHandling,
    expectedMatches: 1, // ValidationError
  },
  {
    id: 'adv-arrow-async',
    category: 'advanced',
    language: 'javascript',
    description: 'Find async arrow functions',
    pattern: 'async ($$$ARGS) => { $$$BODY }',
    codeContext: CODE_CONTEXTS.expressServer,
    expectedMatches: 1,
  },
  {
    id: 'adv-res-json',
    category: 'advanced',
    language: 'javascript',
    description: 'Find Express res.json() calls',
    pattern: 'res.json($$$ARGS)',
    codeContext: CODE_CONTEXTS.expressServer,
    expectedMatches: 3,
  },
];

// ============================================================================
// Exports
// ============================================================================

export const ALL_PATTERNS: AstGrepPattern[] = [...basicPatterns, ...intermediatePatterns, ...advancedPatterns];

export const BASIC_PATTERNS = basicPatterns;
export const INTERMEDIATE_PATTERNS = intermediatePatterns;
export const ADVANCED_PATTERNS = advancedPatterns;

/**
 * Generate a curriculum-ordered dataset (basic → intermediate → advanced)
 */
export function generateCurriculumDataset(numExamples: number): AstGrepPattern[] {
  const result: AstGrepPattern[] = [];

  // Calculate proportions: 40% basic, 35% intermediate, 25% advanced
  const basicCount = Math.floor(numExamples * 0.4);
  const intermediateCount = Math.floor(numExamples * 0.35);
  const advancedCount = numExamples - basicCount - intermediateCount;

  // Shuffle within each category
  const shuffledBasic = [...basicPatterns].sort(() => Math.random() - 0.5);
  const shuffledIntermediate = [...intermediatePatterns].sort(() => Math.random() - 0.5);
  const shuffledAdvanced = [...advancedPatterns].sort(() => Math.random() - 0.5);

  // Fill each category
  for (let i = 0; i < basicCount; i++) {
    result.push(shuffledBasic[i % shuffledBasic.length]);
  }
  for (let i = 0; i < intermediateCount; i++) {
    result.push(shuffledIntermediate[i % shuffledIntermediate.length]);
  }
  for (let i = 0; i < advancedCount; i++) {
    result.push(shuffledAdvanced[i % shuffledAdvanced.length]);
  }

  return result;
}

/**
 * Generate a randomly shuffled dataset
 */
export function generateShuffledDataset(numExamples: number): AstGrepPattern[] {
  const result: AstGrepPattern[] = [];
  const shuffled = [...ALL_PATTERNS].sort(() => Math.random() - 0.5);

  while (result.length < numExamples) {
    result.push(...shuffled);
  }

  return result.slice(0, numExamples);
}
