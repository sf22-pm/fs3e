# FS3E

Arcabouço de software para agrupar, automatizar e simplificar a execução de métodos de seleção de características e as respectivas avaliações utilizando modelos de aprendizado de máquina.

# Como instalar

## Ambiente 

A  FS3E foi desenvolvida em um sistema Linux com as seguintes especificações:
- OS: Linux Ubuntu 22.04 LTS
- Kernel: 5.15.0-41-generic
- Python 3.10.4
- Pip 22.0.2

## Dependências

Basta instalar os módulos Python necessários através do comando:
```bash
$ pip3 install -r requirements.txt
```

# Como executar 

A FS3E possui dois comandos: `run` e `list`. Cada comando possui seu próprio manual de uso, que pode ser visto com a opção `--help` 

```bash
$ python3 fs3e.py run --help
```

```bash
$ python3 fs3e.py list --help
```

Ajuda padrão (saída) da ferramenta:
```bash
Usage: fs3e.py [-h] {list,run} ...

Tool designed to run and evaluate feature selection methods for Android malwares detection

Options:
  -h, --help  show this help message and exit

Available commands:
  {list,run}
    list      List available feature selection methods and/or machine learning models
    run       Run experiment with feature selection methods and ML models

```

## Exemplos de uso 1:
```bash
$ python3 fs3e.py list --methods
$ python3 fs3e.py list --models
$ python3 fs3e.py list --all
$ python3 fs3e.py run -d dataset/drebin_215.csv
```

## Exemplos de uso 2: executando métodos e modelo específicos
```bash
$ python3 fs3e.py run --fs-methods rfg sigapi --ml-model rf -d datasets/*.csv
```

