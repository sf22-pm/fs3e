# SigPID - Significant Permission Identification

This is a system to prunes permissions that have low impacts on detection effectiveness using multilevel data pruning to reduce analysis efforts.

### How to use

NOTE: This procedure has only been tested in Ubuntu 20.04

**0. Requirements**
- Machine Learning Library Extensions (MLXTEND):
    ```sh
    $ sudo pip install mlxtend
    ```

**sigpid.py**
- Usage example (must run from within the `feature_selection` directory):
    ```sh
    $ python3 methods.SigPID.sigpid -d drebin.csv
    ```

- How to use (arguments):
    ```sh
    $ python3 methods.SigPID.sigpid --help
    ```

### Output
- MLDP directory
- pis_PRNR.png
- pis_SPR.png
