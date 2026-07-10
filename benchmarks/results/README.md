# Benchmark results

Benchmark history is generated locally by:

```bash
uv run --only-group bench python -m benchmarks.run_benchmarks
```

The resulting `benchmark_history.parquet` file is intentionally ignored because it is machine-specific, grows over time,
and is rewritten atomically by the benchmark runner. Preserve benchmark datasets as explicitly versioned release
artifacts or attach them to a benchmark report when long-term comparison is required.
