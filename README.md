# ShotgunCSP

**ShotgunCSP** is a Python package designed to solve the crystal structure prediction (CSP) problem using a non-iterative, single-shot screening framework. This method leverages a large library of virtually created crystal structures and employs a machine-learning energy predictor for efficient and accurate predictions.

## Features

- Non-iterative, single-shot screening framework for CSP
- Transfer learning for accurate energy prediction
- Generative models based on element substitution (**ShotgunCSP-GT**) and symmetry-restricted structure generation (**ShotgunCSP-GW**)
- High prediction accuracy with reduced computational intensity

> **NOTE**
>
> The release of the ShotgunCSP-GW-based generator is anticipated for 2026.


## Installation

First, you’ll need to install Rust. Follow the official installation guide to set up the latest Rust toolchain. You can do this by entering the following commands:

```bash
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
$ rustup update
```

The ShotgunCSP package is managed using rye, a robust project and package management tool for Python. To install rye, use the following command:

```bash
$ curl -sSf https://rye.astral.sh/get | bash
```

By default, installing ShotgunCSP will also install a compatible version of PyTorch. If you need a different version of PyTorch, you can remove the default by running rye remove torch, and then install your preferred version, making sure it is compatible with PyTorch ^2.0.0. Refer to the official installation guide for further details.

To install ShotgunCSP, use the following command:

```bash
$ rye sync -f && rye run dev
```

This will create a virtual Python environment in a .ven folder within the root directory and install the development version of ShotgunCSP.

Alternatively, to install an optimized version of ShotgunCSP, use these commands:

```bash
$ rye sync -f && rye run build
$ rye run pip install -U target/wheels/shotgun_csp-0.3.2rc1-*
```

## Usage

Here is a simple example of how to use **shotgun-csp**:

```python
from shotgun_csp.generator import TemplateSelector
from shotgun_csp.utils import VASPInputGenerator

# Select templates
selector = TemplateSelector(target=<composition>, volume=<predicted volume>)
templates = selector(<pymatgen structures>, filter=<structure filter (optional)>)

# Generate VASP input
generator = VASPInputGenerator(save_to='/path/to/save')
generator.static_input(<pymatgen structure>)  # static calculation
generator.relax_input(<pymatgen structure>)  # relax calculation

```

See [example](examples/) for details.

<!-- Please refer to the [documentation](https://yourdocumentationlink) for more detailed instructions and advanced usage. -->

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions to improve **ShotgunCSP**. Please fork the repository and submit your pull requests.

## Acknowledgements

We would like to thank all contributors and the scientific community for their valuable input and support.

## Contact

For any inquiries or issues, please [open an issue](https://github.com/TsumiNa/ShutgunCSP/issues/new/choose).
