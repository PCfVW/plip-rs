# PLIP Corpus

Sample code corpus for Python vs Rust language classification experiments.

## Format

The corpus is a JSON file with the following structure:

```json
{
  "samples": [
    {"language": "python", "code": "..."},
    {"language": "rust", "code": "..."}
  ],
  "metadata": {...}
}
```

## Current Samples

The included `samples.json` is a **minimal seed corpus** with 5 Python and 5 Rust samples for testing.

For real experiments, you should expand this to at least:
- 50 Python samples
- 50 Rust samples

## Sample Selection Guidelines

1. **Equivalent algorithms**: Include the same algorithm implemented in both languages
2. **Idiomatic code**: Use language-specific idioms (e.g., pattern matching in Rust)
3. **Varied complexity**: Mix simple functions with more complex implementations
4. **Data structures**: Include samples using common data structures
5. **Error handling**: Include samples with error handling patterns

## Generating Samples

You can generate additional samples from:
- Open-source projects (with proper licensing)
- Algorithm implementations from textbooks
- Your own code written in both languages

## Balance

Keep the corpus balanced with equal numbers of Python and Rust samples to avoid classifier bias.
