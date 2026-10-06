# Third-party software and provenance

The previous exploratory code included a substantial standalone Mask R-CNN / ResNet-FPN implementation with training utilities. The public portfolio package does **not** copy that training framework or bundle any model checkpoint, data, annotations, or third-party source.

This reconstruction is authored as standalone sample-scoring/selection code. A user may independently integrate:

- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/index.html) for the optional segmentation model and output adapter;
- [NumPy](https://numpy.org/) for mask and geometry operations;
- [pytest](https://pytest.org/) for unit tests.

Check license conditions for actual package versions, model weights and datasets before using or redistributing them. The historical source tree is still accessible through Git history; its upstream provenance and reuse rights have **not** been audited or relicensed by the modern package.

No upstream model paper, library or toolkit should be interpreted as authored by this repository's maintainer. Historical model components and experimental results are distinct from this portable acquisition-code reconstruction.
