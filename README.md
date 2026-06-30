# codif-ape-train 🚀

This repository is designed for training a [`torchTextClassifiers`](https://github.com/InseeFrLab/torchTextClassifiers) model on the NACE dataset. The dataset contains textual data, specifically activity descriptions declared by business owners. The text is labeled in the French version of the NACE (NAF), that contains 732 labels for the 2008 version and 745 for the 2025 one.

## Getting Started

### Launching a Training
To start training the model with default parameters (that are specified in `src/configs`), use the following commands:

```bash
uv run train-ape
```
or similarly:

```bash
uv run -m src.train
```

### Hydra Integration
This project leverages [Hydra](https://hydra.cc/docs/intro/) for configuration management. Hydra allows you to override parameters directly from the command line. For example:

- Override the number of training epochs **and** disable the attention configuration :
  ```bash
  uv run train-ape training_config.num_epochs=1 model_config.attention_config=null
  ```

- Perform a multirun:
  ```bash
  uv run train-ape -m model_config.attention_config.n_layers=2,4,8
  ```

Refer to the [Hydra documentation](https://hydra.cc/docs/intro/) for more details on its capabilities.

## Argo Workflow Template
An Argo template is also available in the `argo-workflows/` directory. This template can be used to orchestrate and automate the training process in a Kubernetes environment.

## License
This project is licensed under the terms of the [LICENSE](./LICENSE) file.
