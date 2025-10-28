Gesture NLP

Implementation of beam search to increase performance of Gesture project.
Using GlossaBART to analyse and compute best glosses.

## Features

- **Hybrid CV + LLM Beam Search**: Combines computer vision probabilities with LLM linguistic coherence
- **Batched Optimization**: ~5x faster inference with batched beam search processing
- **GlossaBART Integration**: Translation from gloss sequences to natural language
- **Attention Analysis**: Semantic compatibility scoring using BART encoder layers

## Quick Test

To quickly display the differences between batched and non-batched, simply run:
```bash
python beam_search.py
```

## Performance

- **Original**: Sequential LLM calls per beam candidate
- **Optimized**: Batched processing