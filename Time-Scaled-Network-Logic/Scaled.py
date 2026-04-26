import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(layout="wide", page_title="Time-Scaled Network Logic")

st.title("PERT/CPM: Time-Scaled Network Logic")
st.markdown("Calcula la ruta crítica y renderiza la red a escala de tiempo. Las tareas miden exactamente $t$ y las conexiones son estrictamente ortogonales.")

# ENTRADA DE DATOS
st.header("1. Ingreso de Actividades")
default_data = [
    {"Actividad": "A", "Predecesores": "", "t": 3.0},
    {"Actividad": "B", "Predecesores": "A", "t": 4.0},
    {"Actividad": "C", "Predecesores": "A", "t": 2.0},
    {"Actividad": "D", "Predecesores": "B", "t": 5.0},
    {"Actividad": "E", "Predecesores": "C", "t": 1.0},
    {"Actividad": "F", "Predecesores": "C", "t": 2.0},
    {"Actividad": "G", "Predecesores": "D, E", "t": 4.0},
    {"Actividad": "H", "Predecesores": "F, G", "t": 3.0},
]

df_inicial = pd.DataFrame(default_data)
# Forzamos formato decimal (float) desde el inicio
df_inicial['t'] = df_inicial['t'].astype(float)
edited_df = st.data_editor(df_inicial, num_rows="dynamic", use_container_width=True)

# MOTOR LÓGICO CPM 
def process_cpm(df):
    if df.empty: return None

    # Limpieza de datos
    df['t'] = pd.to_numeric(df['t'], errors='coerce').fillna(0.0)
    df['Actividad'] = df['Actividad'].astype(str).str.strip()
    df['Predecesores'] = df['Predecesores'].astype(str).fillna('')

    G = nx.DiGraph()
    for idx, row in df.iterrows():
        if row['Actividad']:
            G.add_node(row['Actividad'], t=row['t'])

    for idx, row in df.iterrows():
        act = row['Actividad']
        preds = [p.strip() for p in row['Predecesores'].split(',') if p.strip()]
        for p in preds:
            if p in G.nodes:
                G.add_edge(p, act)

    if not nx.is_directed_acyclic_graph(G):
        return "ERROR_CICLO"

    # Pase hacia Adelante (ES, EF)
    ES, EF = {}, {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        ES[node] = max([EF[p] for p in preds]) if preds else 0.0
        EF[node] = ES[node] + G.nodes[node]['t']

    project_duration = max(EF.values()) if EF else 0.0

    # Pase hacia Atrás (LS, LF)
    LS, LF = {}, {}
    for node in reversed(list(nx.topological_sort(G))):
        succs = list(G.successors(node))
        LF[node] = min([LS[s] for s in succs]) if succs else project_duration
        LS[node] = LF[node] - G.nodes[node]['t']

    # Compilar resultados
    results = []
    for node in G.nodes():
        slack = LS[node] - ES[node]
        results.append({
            'Actividad': node,
            't': G.nodes[node]['t'],
            'ES': ES[node],
            'EF': EF[node],
            'LS': LS[node],
            'LF': LF[node],
            'Holgura': slack,
            'Critica': abs(slack) < 1e-6, # Tolerancia para decimales
            'Predecesores': list(G.predecessors(node))
        })

    return pd.DataFrame(results)

# ALGORITMO DE CARRILES
def assign_lanes(df_cpm):
    # Ordenar tareas por inicio temprano (ES)
    df_sorted = df_cpm.sort_values(by=['ES', 't'], ascending=[True, False])
    lanes = [] # Almacena el tiempo libre de cada carril
    y_coords = {}

    for idx, row in df_sorted.iterrows():
        es, ef = row['ES'], row['EF']
        assigned_lane = -1
        
        # Buscar el primer carril disponible sin colisión (con tolerancia)
        for i, free_time in enumerate(lanes):
            if free_time <= es + 1e-6:
                assigned_lane = i
                lanes[i] = ef
                break
                
        # Crear un nuevo carril
        if assigned_lane == -1:
            assigned_lane = len(lanes)
            lanes.append(ef)
            
        y_coords[row['Actividad']] = assigned_lane

    # Invertir el eje Y para que la primera tarea quede arriba en la gráfica
    max_y = len(lanes) - 1 if lanes else 0
    df_cpm['Y'] = df_cpm['Actividad'].map(lambda x: max_y - y_coords[x])
    
    return df_cpm

# RENDERIZADO CARTESIANO CON PLOTLY
def plot_time_scaled_network(df_cpm):
    fig = go.Figure()
    
    node_coords = {row['Actividad']: (row['EF'], row['Y']) for idx, row in df_cpm.iterrows()}

    # A) Dibujar Conexiones Ortogonales de Precedencia
    for idx, row in df_cpm.iterrows():
        act, es, y_succ = row['Actividad'], row['ES'], row['Y']

        for pred in row['Predecesores']:
            ef_pred, y_pred = node_coords[pred]
            x_path = [ef_pred, ef_pred, es]
            y_path = [y_pred, y_succ, y_succ]

            fig.add_trace(go.Scatter(
                x=x_path, y=y_path, mode='lines',
                line=dict(color='gray', width=1.5, dash='dot'),
                hoverinfo='skip', showlegend=False
            ))

    # B) Dibujar Actividades
    for idx, row in df_cpm.iterrows():
        color = 'red' if row['Critica'] else '#1f77b4' # Rojo para ruta crítica, Azul para el resto
        
        fig.add_trace(go.Scatter(
            x=[row['ES'], row['EF']], y=[row['Y'], row['Y']],
            mode='lines+markers',
            line=dict(color=color, width=5),
            marker=dict(size=12, color=color, symbol='circle'),
            name=row['Actividad'],
            hovertemplate=f"<b>Actividad {row['Actividad']}</b><br>Duración (t): {row['t']:.2f}<br>ES: {row['ES']:.2f}<br>EF: {row['EF']:.2f}<br>Holgura: {row['Holgura']:.2f}<extra></extra>"
        ))
        
        fig.add_annotation(
            x=(row['ES'] + row['EF']) / 2, y=row['Y'],
            text=f"{row['Actividad']} (t={row['t']:.1f})",
            showarrow=False, yshift=15, font=dict(color=color, size=14, family="Arial")
        )
        
        if row['Holgura'] > 1e-6:
            fig.add_trace(go.Scatter(
                x=[row['EF'], row['LF']], y=[row['Y'], row['Y']],
                mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                hoverinfo='skip', showlegend=False
            ))

#MARCADOR DE DURACIÓN TOTAL
    project_duration = df_cpm['EF'].max()
    
    # Línea vertical punteada al final del proyecto
    fig.add_vline(
        x=project_duration, 
        line_width=2, 
        line_dash="dashdot", 
        line_color="green",
        opacity=0.6
    )
    
    
    fig.add_annotation(
        x=project_duration, 
        y=1.05,
        yref="paper",
        text=f"<b>Duración Total max(EF) = {project_duration:.2f}</b>",
        showarrow=False,
        xanchor="center",
        font=dict(size=14, color="green", family="Arial")
    )

    # Configuración Time-Scaled
    fig.update_layout(
        title="Time-Scaled Network Logic Diagram",
        xaxis=dict(title="Tiempo del Proyecto (t)", gridcolor='lightgray', zeroline=True), 
        yaxis=dict(showticklabels=False, gridcolor='white', zeroline=False),
        plot_bgcolor='white',
        height=600,
        showlegend=False,
        hovermode="closest",
        margin=dict(t=80) 
    )
    return fig

# EJECUCIÓN
st.write("---")
if st.button("Generar Diagrama a Escala de Tiempo"):
    df_cpm = process_cpm(edited_df)

    if isinstance(df_cpm, str) and df_cpm == "ERROR_CICLO":
        st.error("Error: Se detectó un ciclo en las dependencias. Revisa la tabla.")
    elif df_cpm is not None and not df_cpm.empty:
        st.success("¡Cálculos CPM completados!")
        
        st.subheader("Tabla de Resultados CPM")
        df_mostrar = df_cpm[['Actividad', 't', 'ES', 'EF', 'LS', 'LF', 'Holgura', 'Critica']]
        
        # Redondeo estético para la tabla
        df_mostrar = df_mostrar.round(2)
        
        # Se usa map() (nueva sintaxis de pandas) o applymap() dependiendo de la versión
        try:
            st.dataframe(df_mostrar.style.map(lambda x: 'background-color: #ffcccc' if x else '', subset=['Critica']), use_container_width=True)
        except AttributeError:
            st.dataframe(df_mostrar.style.applymap(lambda x: 'background-color: #ffcccc' if x else '', subset=['Critica']), use_container_width=True)
        
        df_cpm = assign_lanes(df_cpm)
        fig = plot_time_scaled_network(df_cpm)

        st.subheader("Gráfica a Escala de Tiempo")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("[Guía] Las líneas rojas indican la Ruta Crítica (holgura = 0). Las líneas punteadas grises indican las dependencias ortogonales. Las líneas punteadas de color al final de una tarea indican su holgura disponible.")