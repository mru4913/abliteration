# Abliterate

A Python library for LLM abliteration
    
## Installation (Work in progress)

```bash
pip install abliterate # (FUTURE)
```

## Features

- Model-agnostic design with a focus on extensibility
- Support for various transformer-based language models
- Configurable abliteration parameters
- Built-in dataset management
- Comprehensive test suite
- Pydantic-based configuration validation

## Architecture

The library is organized into several key components:

- `config`: Configuration management using Pydantic
- `core`: Core abliteration engine
- `models`: Model implementations
- `datasets`: Dataset management
- `utils`: Utility functions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Getting Started

### Installation & Project Management

- install uv

```Shell
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
wget -qO- https://astral.sh/uv/install.sh | sh
```

- create virtual environment

```Shell
# create virtual environment
uv venv
uv init # create pyproject.toml
uv lock # create lock file

# install dependencies
uv pip install

# activate virtual environment
source .venv/bin/activate
```

- Project Management
  - `uv init`: Create a new Python project.
  - `uv add`: Add a dependency to the project.
  - `uv remove`: Remove a dependency from the project.
  - `uv add --dev pytest`: To add a development dependency, use the --dev flag:
