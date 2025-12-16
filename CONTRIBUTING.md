# Contributing to MLX-Node

Thank you for your interest in contributing to MLX-Node! We welcome contributions from the community and are excited to work with you.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/mlx-node.git
   cd mlx-node
   ```
3. **Install dependencies**:
   ```bash
   yarn install
   ```
4. **Build the project**:
   ```bash
   yarn build
   ```
5. **Run tests** to ensure everything works:
   ```bash
   yarn test
   ```

## Development Workflow

### Making Changes

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our code style and conventions

3. **Add tests** for new functionality

4. **Run tests** to ensure nothing is broken:

   ```bash
   yarn test
   yarn typecheck
   yarn lint
   ```

5. **Commit your changes** with a descriptive message:
   ```bash
   git commit -m "feat: add new sampling method"
   ```

### Submitting a Pull Request

1. **Push your branch** to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub

3. **Describe your changes** clearly in the PR description

4. **Wait for review** - we'll review your PR as soon as possible

## Code Style Guidelines

### TypeScript

- Use TypedArrays for all numeric data
- Follow the existing patterns in the codebase
- Add JSDoc comments for public APIs
- Use meaningful variable names

### Rust

- Follow Rust naming conventions (snake_case for functions/variables)
- Use `#[napi]` attributes for exports to Node.js
- Add documentation comments with `///`
- Handle errors with `Result<T>` types

### Testing

- Write tests for all new features
- Use descriptive test names
- Test edge cases and error conditions
- Aim for high test coverage

## Adding New Operations

### 1. Core Operations

To add a new MLX operation:

1. Add FFI binding in `mlx-sys/src/lib.rs`
2. Add C++ bridge in `mlx-sys/src/mlx.cpp` if needed
3. Implement Rust wrapper in appropriate module (e.g., `node/src/array.rs`)
4. Run `yarn build` to generate TypeScript definitions
5. Add comprehensive tests

### 2. Neural Network Layers

To add a new layer:

1. Implement in `node/src/nn.rs` with `#[napi]` exports
2. Add gradient computation if needed in `node/src/gradients.rs`
3. Update TypeScript re-exports in `src/nn/`
4. Add tests in `__test__/core/`

### 3. Sampling Methods

To add a new sampling strategy:

1. Implement in `node/src/sampling.rs`
2. Add to the unified `sample` function
3. Update TypeScript types
4. Add tests in `__test__/utils/sampling.test.ts`

## Documentation

- Update relevant markdown files in `docs/`
- Add JSDoc comments for TypeScript APIs
- Add documentation comments for Rust functions
- Update README.md if adding major features

## Performance Considerations

- Use zero-copy operations where possible
- Leverage MLX's lazy evaluation
- Minimize data transfers between JS and Rust
- Profile performance-critical code paths

## Project Structure

```
mlx-node/
├── node/src/           # Rust implementation (NAPI)
├── src/                # TypeScript orchestration
├── __test__/           # Test suite
├── mlx-sys/            # Low-level MLX bindings
└── docs/               # Documentation
```

## Areas for Contribution

We especially welcome contributions in these areas:

- **New Models**: Implement additional model architectures
- **Optimizations**: Performance improvements
- **Documentation**: Improve docs and examples
- **Testing**: Expand test coverage
- **Bug Fixes**: Fix reported issues

## Questions?

If you have questions:

1. Check existing [issues](https://github.com/yourusername/mlx-node/issues)
2. Open a new issue for discussion
3. Join our community discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct

Please note that this project follows a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

Thank you for contributing to MLX-Node! 🚀
