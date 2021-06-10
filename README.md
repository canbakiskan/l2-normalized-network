# CNN with L2 Normalized Weights

In this project I tried to see what effect restricting the filters to be constant l2 norm would have. Also what would happen if we impose Hoyer activation sparsity or orthogonality condition.

## Requirements

Install the requirements:

```bash
pip install -r requirements.txt
```

Then add current directory to `$PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/containing/project/directory"
```

Following commands assume the name of the folder is `l2-normalized-network`.

## Configuration Files

`config.toml` and `parameters.py` contain the hyperparameters and other configuration settings related to the project. Settings that change less frequently are stored in `config.toml`. Settings that change more frequently are stored in `parameters.py` and can be specified as command line arguments when any python file is run.

## License

Apache License 2.0
