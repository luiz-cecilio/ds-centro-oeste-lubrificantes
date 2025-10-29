
import io
import sys
from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="EDA OS — Troca de Óleo",
    page_icon="🛢️",
    layout="wide"
)

st.title("🛢️ EDA — Ordens de Serviço (Troca de Óleo)")
st.caption("Upload um CSV exportado do IBExpert com as colunas de os_basic, os_segmento e clientes unidas por num_os.")

# Centered upload button with custom styling
st.markdown("""
<style>
    .uploadFile {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stFileUploader {
        max-width: 800px;
        margin: 0 auto;
    }
    .stFileUploader > div {
        padding: 2rem;
    }
    .stFileUploader label {
        font-size: 1.2rem !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar — Upload & Options
# -----------------------------
st.sidebar.header("⬆️ Upload & Opções")

delimiter = st.sidebar.selectbox("Delimitador", options=[",",";","\t","|"], index=1, help="Escolha o separador usado ao exportar do IBExpert. Dica: no Brasil é comum usar ';'.")
encoding = st.sidebar.selectbox("Encoding", options=["utf-8","latin1","cp1252"], index=1)
decimal_sep = st.sidebar.selectbox("Separador decimal", options=[",","."], index=1, help="Se seus números vêm com vírgula (ex.: 1.234,56), selecione ','")
date_dayfirst = st.sidebar.checkbox("Datas no formato dia/mês/ano (DD/MM/YYYY)?", value=True)

# Main centered upload area
uploaded = st.file_uploader("📁 Selecione o arquivo CSV com os dados", type=["csv"], help="Arraste o arquivo aqui ou clique para selecionar")

st.sidebar.markdown("---")
st.sidebar.write("**Colunas esperadas (se houver no CSV):**")
expected_cols = [
    "num_os",
    "data_entrada","hora_entrada","data_conclusao","hora_conclusao",
    "total_pecas_bruto","total_pecas_liquido","total_servicos_bruto","total_servicos_liquido",
    "sub_total_bruto","total_liq","nome_cliente",
    "placa","km","fone","celular","nome_fantasia",
]
st.sidebar.code("\n".join(expected_cols))

# -----------------------------
# Helpers
# -----------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip + replace spaces by underscores."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df

def coerce_decimal(series: pd.Series, decimal_sep: str) -> pd.Series:
    """Coerce numeric series that may have comma decimal or thousand separators."""
    s = series.astype(str)
    
    # First, check if values are already in correct format (like 70159.23)
    # If most values have dots and look like proper decimals, treat as already correct
    sample_values = s.dropna().head(100)
    dot_decimal_count = sum(1 for val in sample_values if '.' in val and val.replace('.', '').replace('-', '').isdigit())
    
    if dot_decimal_count > len(sample_values) * 0.5:  # More than 50% are dot-decimal format
        # Values are likely already in correct format (e.g., 70159.23)
        s = s.str.replace("[^0-9.\-]", "", regex=True)
        return pd.to_numeric(s, errors="coerce")
    
    # Original logic for Brazilian format
    if decimal_sep == ",":
        # remove dots as thousand sep, replace comma with dot
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        # remove commas as thousand sep (e.g., 1,234.56 -> 1234.56)
        s = s.str.replace(",", "", regex=False)
    
    # keep only numbers, dot, minus
    s = s.str.replace("[^0-9.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def parse_time_column(s: pd.Series) -> pd.Series:
    """Parse time like 'HH:MM[:SS]' into time. Returns pd.Series of datetime.time or NaT."""
    # handle plain HHMMSS or HH:MM etc.
    return pd.to_datetime(s, errors="coerce").dt.time

def combine_datetime(date_series: pd.Series, time_series: pd.Series, dayfirst: bool=True) -> pd.Series:
    """Combine separate date & time columns into a single datetime. """
    d = pd.to_datetime(date_series, errors="coerce", dayfirst=dayfirst)
    # Convert time_series to string and parse to time if not already
    t = time_series
    if not np.issubdtype(t.dtype, np.datetime64) and t.dtype != "datetime64[ns]":
        t = pd.to_datetime(t, errors="coerce").dt.time
    # Build datetime
    dt = pd.to_datetime(d.astype("datetime64[ns]").dt.date.astype(str) + " " + pd.Series(t).astype(str), errors="coerce")
    return dt

def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().sort_values(ascending=False)
    pct = (miss / len(df) * 100).round(2)
    out = pd.DataFrame({"missing": miss, "missing_%": pct})
    out = out[out["missing"] > 0]
    return out

def infer_client_label(row: pd.Series) -> str:
    """Prefer nome_cliente (da OS), depois nome_fantasia (da tabela clientes)."""
    a = str(row.get("nome_cliente") or "").strip()
    b = str(row.get("nome_fantasia") or "").strip()
    if a:
        return a
    if b:
        return b
    return "—"

def calculate_margin(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate margins for pieces and services."""
    df = df.copy()
    
    # Margem de peças
    if "total_pecas_bruto" in df.columns and "total_pecas_liquido" in df.columns:
        df["margem_pecas"] = ((df["total_pecas_bruto"] - df["total_pecas_liquido"]) / df["total_pecas_bruto"] * 100).fillna(0)
    
    # Margem de serviços
    if "total_servicos_bruto" in df.columns and "total_servicos_liquido" in df.columns:
        df["margem_servicos"] = ((df["total_servicos_bruto"] - df["total_servicos_liquido"]) / df["total_servicos_bruto"] * 100).fillna(0)
    
    # Margem total
    if "sub_total_bruto" in df.columns and "total_liq" in df.columns:
        df["margem_total"] = ((df["sub_total_bruto"] - df["total_liq"]) / df["sub_total_bruto"] * 100).fillna(0)
    
    return df

def has_contact_info(row: pd.Series) -> str:
    """Classify if client has contact info."""
    fone = str(row.get("fone") or "").strip()
    celular = str(row.get("celular") or "").strip()
    
    # Clean contact data - remove patterns like "(  )      -" 
    import re
    fone_clean = re.sub(r'^\(\s*\)\s*-?\s*$', '', fone)
    celular_clean = re.sub(r'^\(\s*\)\s*-?\s*$', '', celular)
    
    # Check if there are actual digits
    fone_valid = bool(re.search(r'\d', fone_clean))
    celular_valid = bool(re.search(r'\d', celular_clean))
    
    if fone_valid and celular_valid:
        return "Ambos"
    elif fone_valid:
        return "Apenas Fone"
    elif celular_valid:
        return "Apenas Celular"
    else:
        return "Sem Contato"

def has_placa(row: pd.Series) -> str:
    """Classify if service has placa (not balcão sale)."""
    placa = str(row.get("placa") or "").strip()
    
    # Clean placa data - remove patterns like "   -", "-", empty spaces
    import re
    placa_clean = re.sub(r'^\s*-?\s*$', '', placa)
    
    # Check if there are actual alphanumeric characters
    placa_valid = bool(re.search(r'[a-zA-Z0-9]', placa_clean))
    
    return "Com Placa" if placa_valid and placa_clean.lower() not in ["nan", "none"] else "Venda Balcão"

# -----------------------------
# Main — Load & Parse
# -----------------------------
if uploaded is None:
    st.info("Faça upload do CSV para iniciar a análise.")
    st.stop()

try:
    df = pd.read_csv(uploaded, delimiter=delimiter, encoding=encoding)
except Exception as e:
    st.error(f"Falha ao ler CSV: {e}")
    st.stop()

df = normalize_columns(df)

# Identify present expected columns
present = [c for c in expected_cols if c in df.columns]
missing = [c for c in expected_cols if c not in df.columns]

if missing:
    st.warning(f"Algumas colunas esperadas não foram encontradas e serão ignoradas: {', '.join(missing)}")

# Try to coerce numeric monetary columns if they exist
money_cols = [
    "total_pecas_bruto","total_pecas_liquido",
    "total_servicos_bruto","total_servicos_liquido",
    "sub_total_bruto","total_liq"
]
for col in money_cols:
    if col in df.columns:
        df[col] = coerce_decimal(df[col], decimal_sep=decimal_sep)

# Try to coerce km as numeric
if "km" in df.columns:
    df["km"] = coerce_decimal(df["km"], decimal_sep=decimal_sep)

# Parse and combine datetimes
if "data_entrada" in df.columns and "hora_entrada" in df.columns:
    df["dt_entrada"] = combine_datetime(df["data_entrada"], df["hora_entrada"], dayfirst=date_dayfirst)
elif "data_entrada" in df.columns:
    df["dt_entrada"] = pd.to_datetime(df["data_entrada"], errors="coerce", dayfirst=date_dayfirst)

if "data_conclusao" in df.columns and "hora_conclusao" in df.columns:
    df["dt_conclusao"] = combine_datetime(df["data_conclusao"], df["hora_conclusao"], dayfirst=date_dayfirst)
elif "data_conclusao" in df.columns:
    df["dt_conclusao"] = pd.to_datetime(df["data_conclusao"], errors="coerce", dayfirst=date_dayfirst)

# Duration in hours and minutes (can be negative/NaT if not concluded)
if "dt_entrada" in df.columns and "dt_conclusao" in df.columns:
    df["duracao_horas"] = (df["dt_conclusao"] - df["dt_entrada"]).dt.total_seconds() / 3600.0
    df["duracao_minutos"] = (df["dt_conclusao"] - df["dt_entrada"]).dt.total_seconds() / 60.0

# Fallback label for cliente
df["cliente_label"] = df.apply(infer_client_label, axis=1)

# Calculate margins
df = calculate_margin(df)

# Add contact and placa classifications
df["tipo_contato"] = df.apply(has_contact_info, axis=1)
df["tipo_servico"] = df.apply(has_placa, axis=1)

# Show data quality info
st.sidebar.markdown("---")
st.sidebar.write("**Qualidade dos Dados:**")
total_rows = len(df)
sem_contato = len(df[df["tipo_contato"] == "Sem Contato"])
venda_balcao = len(df[df["tipo_servico"] == "Venda Balcão"])
st.sidebar.write(f"• Total de registros: {total_rows}")
st.sidebar.write(f"• Sem contato válido: {sem_contato} ({sem_contato/total_rows*100:.1f}%)")
st.sidebar.write(f"• Venda balcão: {venda_balcao} ({venda_balcao/total_rows*100:.1f}%)")

# Check for suspicious monetary values
if "total_liq" in df.columns:
    total_faturamento = df["total_liq"].sum()
    max_valor = df["total_liq"].max()
    valores_altos = len(df[df["total_liq"] > 10000])  # Values > R$ 10k
    ticket_medio_geral = df["total_liq"].mean()
    
    st.sidebar.markdown("---")
    st.sidebar.write("**Verificação Financeira:**")
    st.sidebar.write(f"• Faturamento total: R$ {total_faturamento:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    st.sidebar.write(f"• Maior valor: R$ {max_valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    st.sidebar.write(f"• Ticket médio: R$ {ticket_medio_geral:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    st.sidebar.write(f"• Valores > R$ 10k: {valores_altos}")
    
    # More realistic thresholds
    if total_faturamento > 1000000000:  # > 1 billion
        st.sidebar.warning("⚠️ Faturamento muito alto - verificar normalização!")
    elif max_valor > 100000:  # > R$ 100k
        st.sidebar.warning("⚠️ Valor máximo muito alto - verificar dados!")
    else:
        st.sidebar.success("✅ Valores financeiros parecem normais")

# -----------------------------
# Filters
# -----------------------------
st.subheader("🔎 Filtros")

with st.expander("Mostrar/Ocultar filtros", expanded=True):
    # First row of filters
    col1, col2, col3, col4 = st.columns(4)

    # Date range based on dt_entrada if exists
    if "dt_entrada" in df.columns:
        min_date = pd.to_datetime(df["dt_entrada"]).min()
        max_date = pd.to_datetime(df["dt_entrada"]).max()
    else:
        min_date = None
        max_date = None

    if min_date is not None and not pd.isna(min_date):
        date_range = col1.date_input(
            "Período (pela data de entrada)",
            value=(min_date.date(), max_date.date() if pd.notna(max_date) else min_date.date())
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            mask = (pd.to_datetime(df["dt_entrada"]).dt.date >= start) & (pd.to_datetime(df["dt_entrada"]).dt.date <= end)
            df_f = df.loc[mask].copy()
        else:
            df_f = df.copy()
    else:
        df_f = df.copy()

    # Tipo de serviço filter (Com Placa vs Venda Balcão)
    tipo_servico_options = ["Todos"] + sorted(df_f["tipo_servico"].unique().tolist())
    tipo_servico_sel = col2.selectbox("Tipo de Serviço", options=tipo_servico_options, index=0)
    if tipo_servico_sel != "Todos":
        df_f = df_f[df_f["tipo_servico"] == tipo_servico_sel]

    # Tipo de contato filter
    tipo_contato_options = ["Todos"] + sorted(df_f["tipo_contato"].unique().tolist())
    tipo_contato_sel = col3.selectbox("Tipo de Contato", options=tipo_contato_options, index=0)
    if tipo_contato_sel != "Todos":
        df_f = df_f[df_f["tipo_contato"] == tipo_contato_sel]

    # Quick filter buttons
    col4.write("**Filtros Rápidos:**")
    if col4.button("🚗 Apenas com Placa"):
        df_f = df_f[df_f["tipo_servico"] == "Com Placa"]
    if col4.button("🏪 Apenas Balcão"):
        df_f = df_f[df_f["tipo_servico"] == "Venda Balcão"]
    if col4.button("📞 Com Contato"):
        df_f = df_f[df_f["tipo_contato"] != "Sem Contato"]

    # Second row of filters
    st.markdown("---")
    col5, col6 = st.columns(2)

    # Placa filter (specific plates) - only show valid plates
    import re
    placas_validas = []
    for p in df_f.get("placa", pd.Series(dtype=str)).dropna().astype(str).unique():
        p_clean = re.sub(r'^\s*-?\s*$', '', str(p).strip())
        if p_clean and re.search(r'[a-zA-Z0-9]', p_clean) and p_clean.lower() not in ["nan", "none"]:
            placas_validas.append(p)
    
    placas_validas = sorted(placas_validas)
    placa_sel = col5.multiselect("Placas Específicas", options=placas_validas, default=[])
    if placa_sel:
        df_f = df_f[df_f["placa"].astype(str).isin(placa_sel)]

    # Cliente filter
    clientes = sorted([c for c in df_f.get("cliente_label", pd.Series(dtype=str)).dropna().astype(str).unique() if c != "—"])
    cliente_sel = col6.multiselect("Clientes", options=clientes, default=[])
    if cliente_sel:
        df_f = df_f[df_f["cliente_label"].astype(str).isin(cliente_sel)]

# -----------------------------
# KPIs
# -----------------------------
st.subheader("📊 KPIs principais")

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

total_os = len(df_f)
total_liq_sum = df_f.get("total_liq", pd.Series(dtype=float)).sum(min_count=1)
ticket_medio = total_liq_sum / total_os if total_os and pd.notna(total_liq_sum) else np.nan
dur_med = df_f.get("duracao_horas", pd.Series(dtype=float)).median()
km_med = df_f.get("km", pd.Series(dtype=float)).median()
margem_media = df_f.get("margem_total", pd.Series(dtype=float)).mean()

kpi1.metric("Ordens de Serviço", f"{total_os:,}".replace(",", "."))
kpi2.metric("Faturamento Líquido (Σ)", f"R$ {total_liq_sum:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
kpi3.metric("Ticket Médio", f"R$ {ticket_medio:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
kpi4.metric("Margem Média", "—" if pd.isna(margem_media) else f"{margem_media:.1f}%")
kpi5.metric("Duração Mediana (h)", "—" if pd.isna(dur_med) else f"{dur_med:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
kpi6.metric("KM Mediano", "—" if pd.isna(km_med) else f"{km_med:,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))

# -----------------------------
# Tabs for different analyses
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análise Temporal", "📞 Análise de Contatos", "🚗 Análise de Serviços", "💰 Análise Financeira"])

with tab1:
    st.subheader("📈 Análise Temporal")
    
    # Time series (by month) of total_liq and count
    if "dt_entrada" in df_f.columns:
        ts = df_f.copy()
        ts["mes"] = pd.to_datetime(ts["dt_entrada"]).dt.to_period("M").dt.to_timestamp()
        grp = ts.groupby("mes").agg(
            total_liq=("total_liq","sum"),
            os_qtd=("num_os","count")
        ).reset_index()

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.line(grp, x="mes", y="total_liq", markers=True, title="Faturamento Líquido por Mês")
            fig1.update_layout(height=380, xaxis_title="", yaxis_title="R$")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.bar(grp, x="mes", y="os_qtd", title="Quantidade de OS por Mês")
            fig2.update_layout(height=380, xaxis_title="", yaxis_title="OS")
            st.plotly_chart(fig2, use_container_width=True)

    # Heatmap — Dia da semana x Hora (entrada)
    if "dt_entrada" in df_f.columns:
        st.markdown("**Pico de movimento (Dia da semana × Hora de entrada)**")
        tmp = df_f.copy()
        dt = pd.to_datetime(tmp["dt_entrada"], errors="coerce")
        tmp = tmp.loc[dt.notna()].copy()
        
        if len(tmp) > 0:
            # Use day_of_week (0=Monday, 6=Sunday) and map to Portuguese
            tmp["weekday_num"] = dt.loc[dt.notna()].dt.dayofweek
            tmp["hora"] = dt.loc[dt.notna()].dt.hour
            
            # Map weekday numbers to Portuguese names
            weekday_map = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta", 
                          4: "Sexta", 5: "Sábado", 6: "Domingo"}
            tmp["weekday"] = tmp["weekday_num"].map(weekday_map)
            
            # Create pivot table
            pivot = tmp.pivot_table(index="weekday", columns="hora", values="num_os", aggfunc="count", fill_value=0)
            
            # Reindex to ensure proper order
            desired_order = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
            pivot = pivot.reindex([day for day in desired_order if day in pivot.index])

            if not pivot.empty:
                fig7 = px.imshow(pivot, aspect="auto", title="Heatmap: volume de OS por dia/hora")
                fig7.update_layout(height=420, xaxis_title="Hora do dia", yaxis_title="Dia da semana")
                st.plotly_chart(fig7, use_container_width=True)
            else:
                st.warning("Não há dados suficientes para gerar o heatmap.")
        else:
            st.warning("Não há dados de entrada válidos para gerar o heatmap.")

    # Duration distribution
    if "duracao_minutos" in df_f.columns and df_f["duracao_minutos"].notna().any():
        st.markdown("**Distribuição da duração da OS (minutos)**")
        
        # Filter out unrealistic durations (negative, too long, or zero)
        duration_data = df_f[
            (df_f["duracao_minutos"].notna()) & 
            (df_f["duracao_minutos"] > 0) & 
            (df_f["duracao_minutos"] <= 1440)  # Max 24 hours (1440 minutes)
        ].copy()
        
        if len(duration_data) > 0:
            # Create bins from 0 to 180+ minutes with 10-minute intervals
            bins = list(range(0, 181, 10)) + [float('inf')]
            labels = [f"{i}-{i+9}" for i in range(0, 180, 10)] + ["180+"]
            
            duration_data['duracao_faixa'] = pd.cut(duration_data['duracao_minutos'], 
                                                   bins=bins, labels=labels, right=False)
            
            # Count by bins
            duration_counts = duration_data['duracao_faixa'].value_counts().sort_index()
            
            # Create DataFrame for plotly
            duration_df = pd.DataFrame({
                'Faixa': duration_counts.index,
                'Quantidade': duration_counts.values
            })
            
            fig4 = px.bar(duration_df, x='Faixa', y='Quantidade',
                         title="Distribuição da Duração (0-180+ minutos)")
            fig4.update_layout(height=400, xaxis_title="Duração (minutos)", 
                             yaxis_title="Quantidade de OS",
                             xaxis_tickangle=45)
            st.plotly_chart(fig4, use_container_width=True)
            
            # Show some statistics
            st.markdown("**Estatísticas de Duração:**")
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            stats_col1.metric("Média", f"{duration_data['duracao_minutos'].mean():.1f} min")
            stats_col2.metric("Mediana", f"{duration_data['duracao_minutos'].median():.1f} min")
            stats_col3.metric("Mínimo", f"{duration_data['duracao_minutos'].min():.1f} min")
            stats_col4.metric("Máximo", f"{duration_data['duracao_minutos'].max():.1f} min")
        else:
            st.warning("Não há dados válidos de duração para exibir.")

with tab2:
    st.subheader("📞 Análise de Contatos")
    
    # Contact info summary
    contact_summary = df_f["tipo_contato"].value_counts().reset_index()
    contact_summary.columns = ["Tipo de Contato", "Quantidade de OS"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribuição de Tipos de Contato (por OS)**")
        fig_contact = px.pie(contact_summary, values="Quantidade de OS", names="Tipo de Contato", 
                           title="OS por Tipo de Contato")
        st.plotly_chart(fig_contact, use_container_width=True)
    
    with col2:
        st.markdown("**Resumo de Contatos**")
        total_os = len(df_f)
        com_fone_os = len(df_f[df_f["tipo_contato"].isin(["Apenas Fone", "Ambos"])])
        com_celular_os = len(df_f[df_f["tipo_contato"].isin(["Apenas Celular", "Ambos"])])
        sem_contato_os = len(df_f[df_f["tipo_contato"] == "Sem Contato"])
        
        st.metric("Total de OS", total_os)
        st.metric("OS com Telefone", f"{com_fone_os} ({com_fone_os/total_os*100:.1f}%)")
        st.metric("OS com Celular", f"{com_celular_os} ({com_celular_os/total_os*100:.1f}%)")
        st.metric("OS sem Contato", f"{sem_contato_os} ({sem_contato_os/total_os*100:.1f}%)")
    
    # Unique contacts analysis
    st.markdown("**Análise de Números Únicos**")
    
    import re
    def extract_valid_numbers(df, col_name):
        """Extract valid phone numbers from a column."""
        valid_numbers = set()
        for val in df[col_name].dropna():
            val_str = str(val).strip()
            # Clean and validate
            val_clean = re.sub(r'^\(\s*\)\s*-?\s*$', '', val_str)
            if val_clean and re.search(r'\d', val_clean):
                # Extract just the digits for uniqueness check
                digits_only = re.sub(r'[^\d]', '', val_clean)
                if len(digits_only) >= 8:  # Minimum phone number length
                    valid_numbers.add(digits_only)
        return valid_numbers
    
    fones_unicos = extract_valid_numbers(df_f, "fone") if "fone" in df_f.columns else set()
    celulares_unicos = extract_valid_numbers(df_f, "celular") if "celular" in df_f.columns else set()
    todos_contatos_unicos = fones_unicos.union(celulares_unicos)
    
    col3, col4, col5 = st.columns(3)
    col3.metric("Telefones Únicos", len(fones_unicos))
    col4.metric("Celulares Únicos", len(celulares_unicos))
    col5.metric("Total Contatos Únicos", len(todos_contatos_unicos))
    
    # Revenue by contact type
    if "total_liq" in df_f.columns:
        st.markdown("**Faturamento por Tipo de Contato**")
        revenue_by_contact = df_f.groupby("tipo_contato").agg({
            "total_liq": ["sum", "mean", "count"]
        }).round(2)
        revenue_by_contact.columns = ["Faturamento Total (R$)", "Ticket Médio (R$)", "Quantidade OS"]
        revenue_by_contact = revenue_by_contact.reset_index()
        
        # Format values for better readability
        revenue_by_contact["Faturamento Total (R$)"] = revenue_by_contact["Faturamento Total (R$)"].apply(
            lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
        revenue_by_contact["Ticket Médio (R$)"] = revenue_by_contact["Ticket Médio (R$)"].apply(
            lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
        
        st.dataframe(revenue_by_contact, use_container_width=True)

with tab3:
    st.subheader("🚗 Análise de Serviços")
    
    # Service type summary
    service_summary = df_f["tipo_servico"].value_counts().reset_index()
    service_summary.columns = ["Tipo de Serviço", "Quantidade"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribuição por Tipo de Serviço**")
        fig_service = px.pie(service_summary, values="Quantidade", names="Tipo de Serviço",
                           title="OS por Tipo de Serviço")
        st.plotly_chart(fig_service, use_container_width=True)
    
    with col2:
        st.markdown("**Resumo de Serviços**")
        total_os = len(df_f)
        com_placa = len(df_f[df_f["tipo_servico"] == "Com Placa"])
        venda_balcao = len(df_f[df_f["tipo_servico"] == "Venda Balcão"])
        
        st.metric("Total de OS", total_os)
        st.metric("Com Placa", f"{com_placa} ({com_placa/total_os*100:.1f}%)")
        st.metric("Venda Balcão", f"{venda_balcao} ({venda_balcao/total_os*100:.1f}%)")
    
    # Top placas por receita / quantidade
    if "placa" in df_f.columns:
        st.markdown("**Top Placas**")
        c3, c4 = st.columns(2)
        topn = 15
        base = df_f[df_f["tipo_servico"] == "Com Placa"].copy()
        
        if len(base) > 0:
            with c3:
                grp_p = base.groupby("placa").agg(
                    os_qtd=("num_os","count"),
                    receita=("total_liq","sum")
                ).reset_index().sort_values("receita", ascending=False).head(topn)
                fig5 = px.bar(grp_p, x="placa", y="receita", title=f"Top {topn} placas por Receita")
                fig5.update_layout(height=380, xaxis_title="Placa", yaxis_title="R$")
                st.plotly_chart(fig5, use_container_width=True)

            with c4:
                grp_q = base.groupby("placa").agg(
                    os_qtd=("num_os","count"),
                    receita=("total_liq","sum")
                ).reset_index().sort_values("os_qtd", ascending=False).head(topn)
                fig6 = px.bar(grp_q, x="placa", y="os_qtd", title=f"Top {topn} placas por Quantidade")
                fig6.update_layout(height=380, xaxis_title="Placa", yaxis_title="OS")
                st.plotly_chart(fig6, use_container_width=True)
    
    # Revenue comparison
    if "total_liq" in df_f.columns:
        st.markdown("**Comparação Financeira por Tipo de Serviço**")
        revenue_by_service = df_f.groupby("tipo_servico").agg({
            "total_liq": ["sum", "mean", "count"]
        }).round(2)
        revenue_by_service.columns = ["Faturamento Total", "Ticket Médio", "Quantidade OS"]
        revenue_by_service = revenue_by_service.reset_index()
        
        fig_revenue_service = px.bar(revenue_by_service, x="tipo_servico", y="Faturamento Total",
                                   title="Faturamento Total por Tipo de Serviço")
        st.plotly_chart(fig_revenue_service, use_container_width=True)
        
        st.dataframe(revenue_by_service)

with tab4:
    st.subheader("💰 Análise Financeira")
    
    # Distribution of ticket (total_liq)
    if "total_liq" in df_f.columns:
        st.markdown("**Distribuição do valor líquido da OS**")
        fig3 = px.histogram(df_f, x="total_liq", nbins=50, title="Distribuição dos Valores das OS")
        fig3.update_layout(height=360, xaxis_title="R$ por OS", yaxis_title="Frequência")
        st.plotly_chart(fig3, use_container_width=True)
    
    # Margin analysis
    if "margem_total" in df_f.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distribuição da Margem Total**")
            fig_margin = px.histogram(df_f[df_f["margem_total"].notna()], x="margem_total", nbins=30,
                                    title="Distribuição da Margem (%)")
            fig_margin.update_layout(height=360, xaxis_title="Margem (%)", yaxis_title="Frequência")
            st.plotly_chart(fig_margin, use_container_width=True)
        
        with col2:
            st.markdown("**Margem por Tipo de Serviço**")
            margin_by_service = df_f.groupby("tipo_servico")["margem_total"].mean().reset_index()
            fig_margin_service = px.bar(margin_by_service, x="tipo_servico", y="margem_total",
                                      title="Margem Média por Tipo de Serviço")
            fig_margin_service.update_layout(height=360, xaxis_title="Tipo de Serviço", yaxis_title="Margem Média (%)")
            st.plotly_chart(fig_margin_service, use_container_width=True)
    
    # Financial summary table
    if all(col in df_f.columns for col in ["total_pecas_bruto", "total_servicos_bruto", "sub_total_bruto", "total_liq"]):
        st.markdown("**Resumo Financeiro**")
        financial_summary = pd.DataFrame({
            "Métrica": ["Peças Bruto", "Serviços Bruto", "Sub-total Bruto", "Total Líquido"],
            "Total (R$)": [
                df_f["total_pecas_bruto"].sum(),
                df_f["total_servicos_bruto"].sum(), 
                df_f["sub_total_bruto"].sum(),
                df_f["total_liq"].sum()
            ],
            "Média (R$)": [
                df_f["total_pecas_bruto"].mean(),
                df_f["total_servicos_bruto"].mean(),
                df_f["sub_total_bruto"].mean(),
                df_f["total_liq"].mean()
            ]
        })
        financial_summary = financial_summary.round(2)
        st.dataframe(financial_summary)

# -----------------------------
# Tabelas & Missing
# -----------------------------
st.subheader("🧾 Amostra de dados filtrados")
st.dataframe(df_f.head(100))

st.subheader("⚠️ Valores ausentes")
miss_df = summarize_missing(df_f)
if miss_df.empty:
    st.success("Sem valores ausentes nas colunas presentes.")
else:
    st.dataframe(miss_df)

# -----------------------------
# Download
# -----------------------------
st.subheader("⬇️ Download dos dados filtrados")
@st.cache_data
def to_csv_bytes(_df: pd.DataFrame) -> bytes:
    return _df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Baixar CSV filtrado",
    data=to_csv_bytes(df_f),
    file_name="os_filtrado.csv",
    mime="text/csv"
)

st.caption("💡 Dicas: Verifique se seu export inclui as colunas de data e hora separadas para melhor análise de duração. Ajuste os filtros acima para focar em períodos e clientes específicos.")
