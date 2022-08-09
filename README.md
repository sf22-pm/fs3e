# fs3e

Ferramenta para automatizar e simplificar a execução de métodos de seleção de características (FS) e modelos de machine learning (ML).

Primeiras ideias:
1. A ferramenta deve ser capaz de executar o método selecionado pelo usuário;
2. A ferramenta deve ser flexível e simples para incorporar novos métodos (e.g., 1 diretório e 1 script de bootstrap por método);
3. Com a saída do método (i.e., dataset de características selecionadas), a ferramenta deve executar os modelos RF e SVM;
4. A ferramenta irá apresentar o resultado das métricas dos modelos;
5. A ferramenta deve permitir especificar: 
- o dataset de entrada;
- o prefixo do arquivo do dataset de saída de cada método de seleção;
- o arquivo de saída para as métricas dos modelos de ML RF e SVM;

6. A ferramenta poderá também gerar automaticamente gráficos ou tabelas das saídas dos modelos RF e SVM;

# Como instalar

## Ambiente 

O `fs3e.py` foi desenvolvido em um sistema Linux com as seguintes especificações:
- OS: Linux Ubuntu 22.04 LTS
- Kernel: 5.15.0-41-generic
- Python 3.10.4
- Pip 22.0.2

## Dependências

- O `fs3e.py` em si requer apenas as dependências listadas no arquivo `requirements.txt`, que você pode instalar com o comando `pip3 install -r requirements.txt`. 

- Já os métodos e modelos possuem suas próprias dependências. Elas são verificadas nos shell scripts `run` de cada método e modelo. Se for preciso instalar alguma, isso lhe será informado ao executar a ferramenta.

# Como rodar

O `fs3e.py` possui dois subcomandos: `run` e `list`, como mostra os detalhes de uso a seguir. Cada subcomando possui seu próprio manual de uso, que pode ser visto com a opção `--help` (e.g. `python3 fs3e.py run --help`).

```bash
usage: fs3e.py [-h] {list,run} ...

Suite to run feature selection (FS) methods and evaluation of machine learning (ML) algorithms

options:
  -h, --help  show this help message and exit

Available commands:
  {list,run}
    list      List available feature selection methods and/or machine learning models
    run       Run experiment with feature selection methods and ML models

```

## Exemplos de uso
```bash
python3 fs3e.py list --methods
python3 fs3e.py list --models
python3 fs3e.py list --all

python3 fs3e.py run -d motodroid.csv

## Rodando com métodos e modelo específicos
python3 fs3e.py run --fs-methods rfg sigapi --ml-model rf -d datasets/*.csv
```
###### methods
Códigos dos métodos.

###### datasets 
Datasets construídos para o estudo.

