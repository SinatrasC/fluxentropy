# Fluxentropy

![ce75d326-20db-47c0-af93-655b34408cc4](https://github.com/user-attachments/assets/c212bf62-2a5b-4900-bfd7-7d59a3291b4a)


Fluxentropy is an open-source engine designed to enhance curriculum learning for language models. By leveraging entropy as a metric to organize training data, Fluxentropy aims to improve the efficiency and performance of model training. Built with modularity in mind, the project centers around using pretrained language models (like Hugging Face’s SmolLM) to assign entropy-based characteristics to dataset chunks, potentially streamlining convergence and optimizing training. Fluxentropy is a project stemming from the work done by the opensource community and spearheaded by [xjdr](https://x.com/_xjdr) and [doomslide](https://x.com/doomslide) (aka shrek and frog) on [entropix](https://github.com/xjdr-alt/entropix).

## Features

- **Entropy Characterization**: Fluxentropy’s core module, built on top of Hugging Face tools, enables entropy assessment and tagging of data chunks. The setup is customizable, handling tokenization, encoding, and entropy measurement in a flexible pipeline.
- **Curriculum Learning via Entropy**: For curriculum learning, dataset chunks are ordered by entropy instead of randomly, optimizing the learning progression.
- **Potential for Enhanced RAG**: Though initially focused on training, Fluxentropy’s entropy-based chunking could later enhance retrieval-augmented generation (RAG) tasks by prioritizing high-value chunks.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SinastrasC/fluxentropy.git
   cd fluxentropy
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

### TODOs

- **Core Functionality**: Develop an entropy characterization function based on Hugging Face tools, capable of tagging entropy levels for dataset chunks.
- **Testing with nanoGPT**: Use entropy-ordered data chunks to test training speed and convergence improvements in nanoGPT.
- **Visualization**: Integrate visualization for entropy distribution and training efficiency metrics to optimize curriculum learning.

## Roadmap

1. **Milestone 1**: ~~Build and validate the `entropy_characterize` function to tag entropy levels and output results to a file.~~
3. **Milestone 2**: Implement visualization for entropy-based data preparation and assess improvements in training efficiency.
   - **Sidequest 1**: ~~Implement statistical analysis to gauge entropy-based ordering across models~~
   - **Sidequest 2**: ~~Correllate benchmark Q&A performance with assigned entropy.~~
   - **Sidequest 2**: ~~Create llama3 tokenized fineweb10B dataset for sorting.~~
5. **Milestone 3**: Connect Fluxentropy to a data import pipeline for data scheduling during training.

## Contributing

Collaboration is central to Fluxentropy! We’re focused on core features initially, but plan to bring on new contributors to expand functionality and test the engine across diverse tasks.

---

Join us in making model training smarter, one entropy-ordered chunk at a time!
