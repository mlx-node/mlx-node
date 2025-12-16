# MLX-Node Documentation

Welcome to the MLX-Node documentation! This directory contains detailed technical documentation about the project's architecture, implementation, and development history.

## 📚 Documentation Structure

### Core Documentation

- **[AUTOGRAD_INTEGRATION.md](AUTOGRAD_INTEGRATION.md)** - Complete guide to the autograd (automatic differentiation) system with functional forward pass architecture
- **[FEATURE_ALIGNMENT_SESSION.md](FEATURE_ALIGNMENT_SESSION.md)** - Feature parity implementation details with MLX-LM and TRL
- **[DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md)** - Complete development timeline and major milestones
- **[SAFETENSORS_LOADER.md](SAFETENSORS_LOADER.md)** - SafeTensors format support and loading implementation

### Performance Documentation

- **[PERFORMANCE_OPTIMIZATIONS.md](PERFORMANCE_OPTIMIZATIONS.md)** - Performance optimization techniques and results
- **[PERFORMANCE_INVESTIGATION.md](PERFORMANCE_INVESTIGATION.md)** - Performance analysis and bottleneck identification
- **[PERFORMANCE_FIX.md](PERFORMANCE_FIX.md)** - Specific performance fixes and their impact

### Bug Fixes & Technical Deep Dives

- **[causal-mask-bug-fix.md](causal-mask-bug-fix.md)** - Causal masking implementation fix details
- **[causal-mask-root-cause.md](causal-mask-root-cause.md)** - Root cause analysis of the causal mask issue

### GRPO Training Documentation

- **[grpo/README.md](grpo/README.md)** - GRPO training overview and guide
- **[grpo/algorithm.md](grpo/algorithm.md)** - GRPO algorithm details and variants
- **[grpo/architecture.md](grpo/architecture.md)** - GRPO implementation architecture
- **[grpo/implementation-plan.md](grpo/implementation-plan.md)** - Original implementation plan (historical)
- **[grpo/implementation-review.md](grpo/implementation-review.md)** - Implementation comparison with reference (historical)

## 🚀 Quick Links

### For Users

- [Getting Started](../README.md#quick-start)
- [API Examples](../README.md#basic-usage)
- [Performance Guide](PERFORMANCE_OPTIMIZATIONS.md)

### For Contributors

- [Contributing Guide](../CONTRIBUTING.md)
- [Development History](DEVELOPMENT_HISTORY.md)
- [Architecture Overview](../README.md#architecture)

### For Researchers

- [GRPO Algorithm](grpo/algorithm.md)
- [Autograd System](AUTOGRAD_INTEGRATION.md)
- [Model Architecture](grpo/architecture.md)

## 📊 Project Status

- **Current Version**: Production-ready (January 2025)
- **Test Coverage**: 99.7% (1,036/1,039 tests passing)
- **Feature Parity**: 100% TRL GRPO, 90% MLX-LM
- **Performance**: Within 8% of reference implementation

## 🔍 Key Technical Achievements

1. **Autograd Integration**: Automatic differentiation with functional forward pass
2. **Causal Masking**: Correct autoregressive behavior in all modes
3. **Feature Alignment**: Complete GRPO training infrastructure
4. **Performance**: Optimized to within 8% of MLX-LM reference

## 📝 Documentation Standards

All documentation in this directory follows these standards:

- **Historical Markers**: Documents with outdated information are marked with historical notes
- **Status Indicators**: Clear status markers (✅, ⚠️, ❌) for features and implementations
- **Code Examples**: Practical examples with actual file references
- **Performance Metrics**: Quantified improvements with benchmarks
- **File References**: Direct links to implementation files with line numbers

## 🔗 External References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [HuggingFace TRL](https://github.com/huggingface/trl)
- [NAPI-RS](https://napi.rs/)

---

_Last Updated: January 2025_
