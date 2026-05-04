import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import math

st.set_page_config(layout="wide", page_title="PERT/CPM — Red a Escala de Tiempo")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f7f9fc;
    color: #1f2937;
}

/* Títulos */
h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #111827;
    margin-bottom: 0.3rem !important;
}

h2, h3 {
    color: #374151 !important;
    font-weight: 600 !important;
}

/* Cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem 1.3rem;
    text-align: center;
    transition: box-shadow 0.2s;
}
.metric-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.metric-label {
    font-size: 0.7rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #111827;
}

.metric-sub {
    font-size: 0.8rem;
    color: #9ca3af;
}

/* Ruta crítica */
.path-box {
    background: #ffffff;
    border-left: 4px solid #e63946;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: monospace;
    color: #374151;
}

/* Botón */
.stButton > button {
    background: #111827;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.5rem 2rem;
    border: none;
}
.stButton > button:hover {
    background: #1f2937;
}

/* Info */
.info-banner {
    background: #eef2ff;
    border: 1px solid #e0e7ff;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #4338ca;
}

div[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Encabezado
st.title("PERT / CPM Red a Escala de Tiempo")
st.markdown(
    "Ingresa **tres estimaciones de tiempo** por actividad (optimista · probable · pesimista). ",
    unsafe_allow_html=False
)

# Tabla de entrada
st.markdown("---")
st.header(" Actividades y estimaciones de tiempo")

st.markdown("""
<div class="info-banner">
<b>a</b> = tiempo optimista &nbsp;|&nbsp; <b>m</b> = tiempo más probable &nbsp;|&nbsp; <b>b</b> = tiempo pesimista<br>
Fórmulas PERT: &nbsp; <b>tₑ = (a + 4m + b) / 6</b> &nbsp;·&nbsp; <b>σ = (b − a) / 6</b> &nbsp;·&nbsp; <b>σ² = ((b − a) / 6)²</b>
</div>
""", unsafe_allow_html=True)

default_data = [
    {"Actividad": "A", "Predecesores": "",     "a": 1.0, "m": 3.0, "b": 5.0},
    {"Actividad": "B", "Predecesores": "A",    "a": 2.0, "m": 4.0, "b": 6.0},
    {"Actividad": "C", "Predecesores": "A",    "a": 1.0, "m": 2.0, "b": 3.0},
    {"Actividad": "D", "Predecesores": "B",    "a": 3.0, "m": 5.0, "b": 7.0},
    {"Actividad": "E", "Predecesores": "C",    "a": 0.5, "m": 1.0, "b": 1.5},
    {"Actividad": "F", "Predecesores": "C",    "a": 1.0, "m": 2.0, "b": 3.0},
    {"Actividad": "G", "Predecesores": "D, E", "a": 2.0, "m": 4.0, "b": 6.0},
    {"Actividad": "H", "Predecesores": "F, G", "a": 2.0, "m": 3.0, "b": 4.0},
]

df_inicial = pd.DataFrame(default_data)
for col in ["a", "m", "b"]:
    df_inicial[col] = df_inicial[col].astype(float)

edited_df = st.data_editor(
    df_inicial, num_rows="dynamic", use_container_width=True,
    column_config={
        "a": st.column_config.NumberColumn("a (optimista)", min_value=0.0, format="%.2f"),
        "m": st.column_config.NumberColumn("m (probable)",  min_value=0.0, format="%.2f"),
        "b": st.column_config.NumberColumn("b (pesimista)", min_value=0.0, format="%.2f"),
    }
)


def process_pert(df: pd.DataFrame):
    if df.empty:
        return None

    for col in ["a", "m", "b"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["Actividad"]    = df["Actividad"].astype(str).str.strip()
    df["Predecesores"] = df["Predecesores"].astype(str).fillna("")

    
    df["te"]    = (df["a"] + 4 * df["m"] + df["b"]) / 6
    df["sigma"] = (df["b"] - df["a"]) / 6
    df["var"]   = df["sigma"] ** 2

    
    G = nx.DiGraph()
    for _, row in df.iterrows():
        if row["Actividad"]:
            G.add_node(row["Actividad"], te=row["te"], sigma=row["sigma"], var=row["var"],
                       a=row["a"], m=row["m"], b=row["b"])

    for _, row in df.iterrows():
        act   = row["Actividad"]
        preds = [p.strip() for p in row["Predecesores"].split(",") if p.strip()]
        for p in preds:
            if p in G.nodes:
                G.add_edge(p, act)

    if not nx.is_directed_acyclic_graph(G):
        return "ERROR_CICLO"

    
    ES, EF = {}, {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        ES[node] = max([EF[p] for p in preds]) if preds else 0.0
        EF[node] = ES[node] + G.nodes[node]["te"]

    project_duration = max(EF.values()) if EF else 0.0


    LS, LF = {}, {}
    for node in reversed(list(nx.topological_sort(G))):
        succs = list(G.successors(node))
        LF[node] = min([LS[s] for s in succs]) if succs else project_duration
        LS[node] = LF[node] - G.nodes[node]["te"]

    
    sources = [n for n in G.nodes if G.in_degree(n) == 0]
    sinks   = [n for n in G.nodes if G.out_degree(n) == 0]

    all_paths = []
    for src in sources:
        for snk in sinks:
            for path in nx.all_simple_paths(G, src, snk):
                length = sum(G.nodes[n]["te"] for n in path)
                all_paths.append((length, path))

    critical_length = max(l for l, _ in all_paths) if all_paths else 0.0
    critical_paths  = [path for l, path in all_paths if abs(l - critical_length) < 1e-6]
    critical_nodes  = set(n for path in critical_paths for n in path)

    
    primary_path = critical_paths[0] if critical_paths else []
    var_project  = sum(G.nodes[n]["var"] for n in primary_path)
    sigma_proj   = math.sqrt(var_project)

    
    rows = []
    for node in G.nodes():
        slack    = round(LS[node] - ES[node], 8)
        critica  = node in critical_nodes
        n_data   = G.nodes[node]
        rows.append({
            "Actividad":    node,
            "a":            n_data["a"],
            "m":            n_data["m"],
            "b":            n_data["b"],
            "tₑ":           n_data["te"],
            "σ":            n_data["sigma"],
            "σ²":           n_data["var"],
            "ES":           ES[node],
            "EF":           EF[node],
            "LS":           LS[node],
            "LF":           LF[node],
            "Holgura":      max(slack, 0.0),
            "Crítica":      critica,
            "Predecesores": list(G.predecessors(node)),
        })

    df_result = pd.DataFrame(rows)
    return df_result, project_duration, critical_paths, sigma_proj, var_project

# Algoritmo
def assign_lanes(df_cpm: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df_cpm.sort_values(["ES", "tₑ"], ascending=[True, False])
    lanes     = []
    y_coords  = {}

    for _, row in df_sorted.iterrows():
        es, ef = row["ES"], row["EF"]
        assigned = -1
        for i, free_at in enumerate(lanes):
            if free_at <= es + 1e-6:
                assigned  = i
                lanes[i]  = ef
                break
        if assigned == -1:
            assigned = len(lanes)
            lanes.append(ef)
        y_coords[row["Actividad"]] = assigned

    max_y = len(lanes) - 1 if lanes else 0
    df_cpm["Y"] = df_cpm["Actividad"].map(lambda x: max_y - y_coords.get(x, 0))
    return df_cpm

# Gráfica a escala de tiempo
def plot_time_scaled(df_cpm: pd.DataFrame, project_duration: float) -> go.Figure:
    fig = go.Figure()

    
    COLOR_CRIT  = "#dc2626"   # rojo profesional
    COLOR_NORM  = "#2563eb"   # azul limpio
    COLOR_CONN  = "#d1d5db"   # gris claro
    COLOR_SLACK = "#7c3aed"   # morado elegante
    BG          = "#f7f9fc"   # fondo general

    
    node_coords = {
        row["Actividad"]: (row["ES"], row["EF"], row["Y"], row["LF"])
        for _, row in df_cpm.iterrows()
    }

    
    for _, row in df_cpm.iterrows():
        act = row["Actividad"]
        es_succ, _, y_succ, _ = node_coords[act]

        for pred in row["Predecesores"]:
            if pred not in node_coords:
                continue

            _, _, y_pred, lf_pred = node_coords[pred]

            
            if lf_pred <= es_succ:
                x_path = [lf_pred, lf_pred, es_succ]
                y_path = [y_pred,  y_succ,  y_succ]
            else:
                x_path = [lf_pred, lf_pred]
                y_path = [y_pred,  y_succ]

            fig.add_trace(go.Scatter(
                x=x_path,
                y=y_path,
                mode="lines",
                line=dict(color=COLOR_CONN, width=1.5, dash="dot"),
                hoverinfo="skip",
                showlegend=False
            ))

    
    for _, row in df_cpm.iterrows():
        act = row["Actividad"]
        es, ef, y, lf = node_coords[act]

        color = COLOR_CRIT if row["Crítica"] else COLOR_NORM
        te = row["tₑ"]
        holgura = row["Holgura"]

        # Línea principal
        fig.add_trace(go.Scatter(
            x=[es, ef],
            y=[y, y],
            mode="lines",
            line=dict(color=color, width=6),
            hovertemplate=(
                f"<b>Actividad {act}</b><br>"
                f"tₑ = {te:.2f} | σ = {row['σ']:.2f}<br>"
                f"ES = {es:.2f}  EF = {ef:.2f}<br>"
                f"LS = {row['LS']:.2f}  LF = {row['LF']:.2f}<br>"
                f"Holgura = {holgura:.2f}"
                "<extra></extra>"
            ),
            showlegend=False
        ))

        # Nodo inicio
        fig.add_trace(go.Scatter(
            x=[es], y=[y],
            mode="markers",
            marker=dict(
                size=10,
                color=color,
                symbol="circle",
                line=dict(color="#ffffff", width=2)
            ),
            hoverinfo="skip",
            showlegend=False
        ))

        # Nodo fin
        fig.add_trace(go.Scatter(
            x=[ef], y=[y],
            mode="markers",
            marker=dict(
                size=10,
                color=color,
                symbol="circle",
                line=dict(color="#ffffff", width=2)
            ),
            hoverinfo="skip",
            showlegend=False
        ))

        # Etiqueta
        fig.add_annotation(
            x=(es + ef) / 2,
            y=y,
            text=f"<b>{act}</b> (tₑ={te:.1f})",
            showarrow=False,
            yshift=18,
            font=dict(color=color, size=12, family="JetBrains Mono"),
        )

        # Holgura
        if holgura > 1e-6:
            fig.add_trace(go.Scatter(
                x=[ef, row["LF"]],
                y=[y, y],
                mode="lines",
                line=dict(color=COLOR_SLACK, width=2.5, dash="dash"),
                hovertemplate=f"Holgura {act}: {holgura:.2f}<extra></extra>",
                showlegend=False
            ))

    
    fig.add_vline(
        x=project_duration,
        line_width=2,
        line_dash="dashdot",
        line_color="#f59e0b",
        opacity=0.8
    )

    fig.add_annotation(
        x=project_duration,
        y=1.06,
        yref="paper",
        text=f"<b>Duración Total = {project_duration:.2f}</b>",
        showarrow=False,
        xanchor="center",
        font=dict(size=13, color="#b45309", family="Inter"),
    )

    
    for label, color in [
        ("Crítica", COLOR_CRIT),
        ("No crítica", COLOR_NORM),
        ("Holgura (al LF)", COLOR_SLACK),
        ("Dependencia ortogonal", COLOR_CONN)
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(
                color=color,
                width=3,
                dash="dash" if "Holgura" in label
                else "dot" if "Dependencia" in label
                else "solid"
            ),
            name=label
        ))

    
    fig.update_layout(
        title=dict(
            text="Red a Escala de Tiempo (Time-Scaled Network Logic)",
            font=dict(family="Inter", size=18, color="#111827")
        ),

        xaxis=dict(
            title="Tiempo del Proyecto (tₑ)",
            gridcolor="#e5e7eb",
            zeroline=True,
            zerolinecolor="#d1d5db",
            tickfont=dict(family="JetBrains Mono", color="#374151"),
            title_font=dict(color="#374151")
        ),

        yaxis=dict(
            showticklabels=False,
            gridcolor="#f1f5f9",
            zeroline=False
        ),

        plot_bgcolor="#ffffff",
        paper_bgcolor=BG,

        font=dict(color="#1f2937"),

        height=620,
        showlegend=True,

        legend=dict(
            bgcolor="#ffffff",
            bordercolor="#e5e7eb",
            borderwidth=1,
            font=dict(family="JetBrains Mono", size=11, color="#374151"),
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top"
        ),

        hovermode="closest",
        margin=dict(t=90, l=20, r=20, b=60),
    )

    return fig


# 
st.markdown("---")
col_btn, _ = st.columns([1, 4])
with col_btn:
    run = st.button("▶ Calcular y Graficar")

if run:
    resultado = process_pert(edited_df.copy())

    if isinstance(resultado, str) and resultado == "ERROR_CICLO":
        st.error("⚠️ Se detectó un ciclo en las dependencias. Revisa la tabla de actividades.")

    elif resultado is not None:
        df_cpm, proj_dur, critical_paths, sigma_proj, var_proj = resultado

        
        st.header("Resultados del Proyecto")

        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (c1, "Duración del Proyecto", f"{proj_dur:.2f}", "max(EF) en ruta crítica"),
            (c2, "Desv. Estándar σ",      f"{sigma_proj:.3f}", "√(Σσ² ruta crítica)"),
            (c3, "Varianza σ²",           f"{var_proj:.3f}", "Σσ² ruta crítica"),
            (c4, "Rutas Críticas",        str(len(critical_paths)), "caminos con holgura = 0"),
        ]
        for col, label, value, sub in metrics:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        #Ruta(s) crítica(s)
        st.markdown("#### Ruta(s) Crítica(s)")
        for i, path in enumerate(critical_paths, 1):
            path_str = " → ".join(path)
            st.markdown(f'<div class="path-box">{"RC " + str(i) if len(critical_paths) > 1 else "Ruta Crítica"}&nbsp;&nbsp;{path_str}</div>',
                        unsafe_allow_html=True)

        # Tabla de resultados
        st.markdown("---")
        st.header("Tabla de Resultados CPM/PERT")

        df_show = df_cpm[[
            "Actividad", "a", "m", "b", "tₑ", "σ", "σ²",
            "ES", "EF", "LS", "LF", "Holgura", "Crítica"
        ]].copy()

        numeric_cols = ["a", "m", "b", "tₑ", "σ", "σ²", "ES", "EF", "LS", "LF", "Holgura"]
        df_show[numeric_cols] = df_show[numeric_cols].round(3)

        def style_row(row):
            if row["Crítica"]:
                return ["background-color: #fee2e2; color: #991b1b"] * len(row)
            return [""] * len(row)

        styled = (
            df_show.style
            .apply(style_row, axis=1)
            .format({c: "{:.3f}" for c in numeric_cols})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Gráfica
        st.markdown("---")
        st.header("Red a Escala de Tiempo")

        df_cpm = assign_lanes(df_cpm)
        fig    = plot_time_scaled(df_cpm, proj_dur)
        st.plotly_chart(fig, use_container_width=True)

        #Guía
        st.markdown("""
        <div class="info-banner">
        🔴 <b>Líneas rojas</b> → Actividades en la Ruta Crítica (Holgura = 0) &nbsp;|&nbsp;
        🔵 <b>Líneas azules</b> → Actividades no críticas &nbsp;|&nbsp;
        🟣 <b>Líneas punteadas moradas</b> → Holgura disponible (terminan exactamente en LF) &nbsp;|&nbsp;
        ⬛ <b>Líneas grises punteadas</b> → Relaciones de precedencia ortogonales (conectan el LF con el sucesor) &nbsp;|&nbsp;
        🟡 <b>Línea amarilla vertical</b> → Duración total del proyecto
        </div>
        """, unsafe_allow_html=True)