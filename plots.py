# plots.py
import altair as alt
import pandas as pd
from utils import create_heatmap, add_label

def plot_counter(counter):
    """
    Cria gráfico de barras Altair com contagem de objetos
    """
    if not counter:
        return None

    df = pd.DataFrame(counter.items(), columns=["Object", "Count"])
    chart = (
        alt.Chart(df)
        .mark_bar(color="#4DA3FF")
        .encode(
            x=alt.X("Object:N", sort="-y"),
            y="Count:Q"
        )
        .properties(width=350, height=300, title="Object count")
    )
    return chart

def build_heatmaps(heatmaps):
    """
    Cria imagens de heatmaps com gradiente 'hot' laranja → vermelho → branco
    e adiciona labels.
    """
    imgs = []
    for label, mask in heatmaps.items():
        heat = create_heatmap(mask)  # usa matplotlib hot
        heat = add_label(heat, label)
        imgs.append(heat)
    return imgs
