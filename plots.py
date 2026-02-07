# plots.py
import altair as alt
import pandas as pd
from utils import create_heatmap, add_label

import altair as alt
import pandas as pd

def plot_counter(counter):
    if not counter:
        return None

    df = pd.DataFrame(counter.items(), columns=["Object", "Count"])

    bar_size = 20  # controla espessura no mobile

    bars = (
        alt.Chart(df)
        .mark_bar(color="#4DA3FF", size=bar_size)
        .encode(
            x=alt.X("Object:N", sort="-y", title=None),
            y=alt.Y("Count:Q", title="Count"),
            tooltip=["Object", "Count"]
        )
    )

    # texto no meio da barra
    text = (
        alt.Chart(df)
        .mark_text(
            dy=-6,              # posição vertical do texto
            color="black",
            fontSize=12
        )
        .encode(
            x="Object:N",
            y="Count:Q",
            text="Count:Q"
        )
    )

    chart = (
        (bars + text)
        .properties(
            width=350,
            height=300,
            title="Object count"
        )
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
