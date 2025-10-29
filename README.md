# 🛢️ EDA OS — Troca de Óleo

Dashboard interativo para análise exploratória de dados (EDA) de Ordens de Serviço relacionadas a trocas de óleo.

## 📋 Descrição

Este aplicativo Streamlit permite fazer upload de arquivos CSV exportados do IBExpert e realizar análises completas incluindo:

- 📈 **Análise Temporal**: Faturamento mensal, quantidade de OS, heatmaps de movimento
- 📞 **Análise de Contatos**: Distribuição de tipos de contato, números únicos
- 🚗 **Análise de Serviços**: Com placa vs Venda balcão, top placas
- 💰 **Análise Financeira**: Distribuição de valores, margens, resumo financeiro

## 🚀 Deploy no Streamlit Community Cloud

### Estrutura do Projeto

```
streamlit/
├── .streamlit/
│   └── config.toml
├── app_oil_eda_v2.py
├── requirements.txt
└── README.md
```

### Como fazer o deploy

1. **Fork ou clone este repositório** no GitHub
2. **Acesse** [share.streamlit.io](https://share.streamlit.io)
3. **Faça login** com sua conta GitHub
4. **Clique em "New app"**
5. **Selecione**:
   - Repository: seu repositório
   - Branch: main (ou master)
   - Main file path: `streamlit/app_oil_eda_v2.py`
6. **Clique em "Deploy!"**

O Streamlit Community Cloud irá automaticamente:
- Detectar o `requirements.txt` e instalar as dependências
- Carregar as configurações do `.streamlit/config.toml`
- Executar sua aplicação

## 📦 Dependências

As dependências estão listadas no `requirements.txt`:
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.14.0

## 📊 Formato dos Dados

O CSV deve conter as seguintes colunas (se disponíveis):

- `num_os`: Número da ordem de serviço
- `data_entrada`, `hora_entrada`: Data e hora de entrada
- `data_conclusao`, `hora_conclusao`: Data e hora de conclusão
- `total_pecas_bruto`, `total_pecas_liquido`: Valores de peças
- `total_servicos_bruto`, `total_servicos_liquido`: Valores de serviços
- `sub_total_bruto`, `total_liq`: Totais
- `nome_cliente`, `nome_fantasia`: Informações do cliente
- `placa`, `km`: Informações do veículo
- `fone`, `celular`: Informações de contato

## 🎨 Configurações

O arquivo `.streamlit/config.toml` contém:
- **Tema personalizado**: Cores e fontes
- **Configurações do servidor**: Tamanho máximo de upload (200MB)

## 📝 Uso Local

Para testar localmente antes do deploy:

```bash
cd streamlit
streamlit run app_oil_eda_v2.py
```

## 📚 Documentação

- [Streamlit Docs](https://docs.streamlit.io)
- [Deploy Guide](https://docs.streamlit.io/deploy/streamlit-community-cloud)

