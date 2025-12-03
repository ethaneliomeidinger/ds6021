import json
import os
os.environ["DGL_GRAPHBOLT_LOAD"] = "0" 
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go

import benchmark as bm

# NEW: PyTorch + DGL-based stuff
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from encoders import GraphSAGEEncoder, DummyEncoder
from hyperfusion import HyperCoCoFusion
from datasets import MultimodalDGLDataset, multimodal_dgl_collate_fn



class GraphSageWrapper(nn.Module):
    def __init__(self, in_feats, hidden_dim=64, out_dim=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.gnn = GraphSAGEEncoder(
            in_feats=in_feats,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, g):
        # Node features were stored in dataset.build_graph as ndata['feat']
        x = g.ndata['feat']  # (total_nodes_in_batch, in_feats)
        return self.gnn(g, x)  # returns (batch_size, out_dim) via dgl.mean_nodes



def run_all_models(
    data_root: str,
    val_frac: float,
    test_frac: float,
    seed: int,
    lasso_alpha: float,
    svr_C: float,
    svr_epsilon: float,
    knn_k: int,
    kmeans_k: int,
):
    """
    Orchestrate all sklearn baselines on concatenated features.

      - load_concatenated_features
      - make_splits
      - run_lasso
      - run_svr
      - run_knn
      - run_kmeans_regression

    Returns:
      results: dict[model_name -> result_dict]
      X_shape: tuple
      y_shape: tuple
    """
    np.random.seed(seed)

    # 1) Load HCP-D (simulated) features
    X, y = bm.load_concatenated_features(data_root)

    # 2) Split into train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = bm.make_splits(
        X, y, val_size=val_frac, test_size=test_frac, seed=seed
    )

    results = {}

    # 3) Lasso
    results["lasso"] = bm.run_lasso(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        alpha=lasso_alpha,
    )

    # 4) SVR
    results["svr"] = bm.run_svr(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        C=svr_C,
        epsilon=svr_epsilon,
        kernel="rbf",
    )

    # 5) KNN
    results["knn"] = bm.run_knn(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        n_neighbors=knn_k,
        weights="distance",
    )

    # 6) KMeans cluster-mean regressor
    results["kmeans"] = bm.run_kmeans_regression(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        n_clusters=kmeans_k,
        random_state=seed,
    )

    return results, X.shape, y.shape


# -----------------------------------------------------------
# Helper: load raw modality npys (for HyperCoCoFusion demo)
# -----------------------------------------------------------
def load_modalities(data_root: str):
    """
    Load fc, sc, morph, cog, labels from data_root/simulated_data-style folder.
    Returns tensors on CPU.
    """
    def _np(path):
        arr = np.load(path)
        return arr

    fc = _np(os.path.join(data_root, "fc.npy"))       # (N, D_fc, D_fc)
    sc = _np(os.path.join(data_root, "sc.npy"))       # (N, D_sc, D_sc)
    morph = _np(os.path.join(data_root, "morph.npy")) # (N, ..., ...)
    cog = _np(os.path.join(data_root, "cog.npy"))     # (N, d_cog)
    labels = _np(os.path.join(data_root, "labels.npy"))

    # Coerce labels to (N, C)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    # Convert to float tensors
    fc_t = torch.from_numpy(fc).float()
    sc_t = torch.from_numpy(sc).float()
    morph_t = torch.from_numpy(morph).float()
    cog_t = torch.from_numpy(cog).float()
    labels_t = torch.from_numpy(labels).float()

    # Flatten morph if it's 3D so MLP sees (N, D)
    N = morph_t.shape[0]
    if morph_t.ndim > 2:
        morph_flat = morph_t.view(N, -1)
    else:
        morph_flat = morph_t

    return fc_t, sc_t, morph_flat, cog_t, labels_t


def run_hypergraph_demo(data_root: str):
    """
    Single forward pass of HyperCoCoFusion using your MultimodalDGLDataset
    (DGL graphs for FC/SC + morph + cog + labels).

    Returns metrics + a small preview of predictions vs labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build dataset & one DataLoader
    ds = MultimodalDGLDataset(root_dir=data_root, threshold=0.0, mode='topk', k=30)
    batch_size = min(32, len(ds))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multimodal_dgl_collate_fn,
        num_workers=0,
    )

    # Take the first batch
    g_fc, g_sc, morph, cog, labels = next(iter(dl))
    g_fc = g_fc.to(device)
    g_sc = g_sc.to(device)
    morph = morph.to(device)   # (B, D, d)
    cog = cog.to(device)       # (B, l)
    labels = labels.to(device) # (B, c)

    B = labels.size(0)
    d_cog = cog.size(-1)
    label_dim = labels.size(-1)

    # 2) Build encoders
    cmsi_dim = 32

    # Node feature dimension (we stored conn_matrix as node feature in dataset.py)
    in_feats = g_fc.ndata["feat"].shape[-1]

    encoder_fc = GraphSageWrapper(
        in_feats=in_feats,
        hidden_dim=64,
        out_dim=cmsi_dim,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    encoder_sc = GraphSageWrapper(
        in_feats=in_feats,
        hidden_dim=64,
        out_dim=cmsi_dim,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    # Morph encoder: use DummyEncoder on a flattened morph vector
    morph_flat = morph.view(B, -1)  # (B, D*d)
    encoder_morph = DummyEncoder(
        in_dim=morph_flat.shape[-1],
        out_dim=cmsi_dim,
    ).to(device)

    # 3) Build HyperCoCoFusion model reusing your existing hyperfusion.py
    model = HyperCoCoFusion(
        encoder_fc=encoder_fc,
        encoder_sc=encoder_sc,
        encoder_morph=encoder_morph,
        cmsi_d=cmsi_dim,
        vwv_d=d_cog,
        label_dim=label_dim,
        cmsi_method="cross_attention",
        info_theory=False,
        vwv_riemann=False,
        use_spectral_hypergraph=True,
        use_residual=True,
    ).to(device)

    # 4) Forward pass
    model.eval()
    with torch.no_grad():
        preds = model(g_fc, g_sc, morph_flat, cog)  # (B, c)

    # 5) Compute simple regression metrics
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except ValueError:
        r2 = float("nan")

    metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "batch_size": int(B),
        "cmsi_dim": int(cmsi_dim),
        "cog_dim": int(d_cog),
        "label_dim": int(label_dim),
        "device": str(device),
    }

    # Preview first few rows
    preview = []
    for i in range(min(5, B)):
        preview.append(
            {
                "idx": int(i),
                "y_true": y_true[i].tolist(),
                "y_pred": y_pred[i].tolist(),
            }
        )

    return metrics, preview


import torch.nn as nn


# -----------------------------------------------------------
# Dash app
# -----------------------------------------------------------

app = Dash(__name__)
# Expose underlying Flask server for Gunicorn / hosting platforms
server = app.server
app.title = "HCP-D Brain Imaging Benchmarks"


app.layout = html.Div(
    style={"margin": "20px", "fontFamily": "Arial, sans-serif"},
    children=[
        html.H2("HCP-D Brain Imaging Benchmark Dashboard"),
        dcc.Tabs(
            id="tabs",
            value="tab-readme",
            children=[
                dcc.Tab(label="README", value="tab-readme"),
                dcc.Tab(label="Baseline Benchmarks", value="tab-benchmarks"),
                dcc.Tab(label="Hypergraph Fusion Demo", value="tab-hypergraph"),
            ],
        ),
        html.Div(id="tab-content"),
    ],
)


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
)
def render_tab(tab):
    if tab == "tab-readme":
        return readme_layout()
    elif tab == "tab-benchmarks":
        return benchmark_layout()
    elif tab == "tab-hypergraph":
        return hypergraph_layout()
    return html.Div("Unknown tab selected.")


# -----------------------------------------------------------
# README tab
# -----------------------------------------------------------
def readme_layout():
    return html.Div(
        style={"maxWidth": "900px"},
        children=[
            html.H3("Overview"),
            html.P(
                "This Dash app runs simple regression baselines on multi-modal "
                "HCP-D brain imaging features (simulated here due to NDA). "
                "Features from cognition, morphology, structural connectivity, "
                "and functional connectivity are concatenated and used to predict "
                "behavioral / clinical targets (e.g., CBCL)."
            ),
            html.H4("Inputs"),
            html.Ul(
                [
                    html.Li("cog.npy: cognitive / behavioral features"),
                    html.Li("morph.npy: cortical morphometry features"),
                    html.Li("sc.npy: structural connectivity features"),
                    html.Li("fc.npy: functional connectivity features"),
                    html.Li("labels.npy: target scores (N or N × T)"),
                ]
            ),
            html.H4("Models Included"),
            html.Ul(
                [
                    html.Li("Lasso (multi-output regression via MultiOutputRegressor)"),
                    html.Li("Support Vector Regression (SVR; RBF kernel)"),
                    html.Li("K-Nearest Neighbors regression"),
                    html.Li("K-Means cluster-mean regression (unsupervised features + cluster-wise label means)"),
                    html.Li("HyperCoCoFusion (spectral hypergraph + CMSI, demo tab)"),
                ]
            ),
            html.P(
                "You can plug in new methods by adding a new run_XXX function in "
                "benchmark.py and registering it in the Dash callbacks."
            ),
        ],
    )


# -----------------------------------------------------------
# Benchmark tab layout
# -----------------------------------------------------------
def benchmark_layout():
    return html.Div(
        style={"maxWidth": "1100px"},
        children=[
            html.H3("Run Baseline Models on HCP-D (Simulated)"),

            html.H4("Data & Split Settings"),
            html.Div(
                style={"display": "flex", "gap": "30px", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        style={"flex": "1 1 250px"},
                        children=[
                            html.Label("Data root (folder with cog.npy, morph.npy, sc.npy, fc.npy, labels.npy)"),
                            dcc.Input(
                                id="data-root",
                                type="text",
                                value="simulated_data",  # default: repo folder
                                style={"width": "100%"},
                            ),
                            html.Small(
                                "Update this path if your .npy files live somewhere else (relative to app root).",
                                style={"display": "block", "marginTop": "4px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1 1 150px"},
                        children=[
                            html.Label("Validation fraction"),
                            dcc.Slider(
                                id="val-frac",
                                min=0.05,
                                max=0.3,
                                step=0.05,
                                value=0.15,
                                marks={float(v): f"{int(v*100)}%" for v in np.arange(0.05, 0.35, 0.05)},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1 1 150px"},
                        children=[
                            html.Label("Test fraction"),
                            dcc.Slider(
                                id="test-frac",
                                min=0.05,
                                max=0.3,
                                step=0.05,
                                value=0.15,
                                marks={float(v): f"{int(v*100)}%" for v in np.arange(0.05, 0.35, 0.05)},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1 1 120px"},
                        children=[
                            html.Label("Random seed"),
                            dcc.Input(
                                id="seed",
                                type="number",
                                value=0,
                                style={"width": "100%"},
                            ),
                        ],
                    ),
                ],
            ),

            html.Hr(),

            html.H4("Model Hyperparameters"),
            html.Div(
                style={"display": "flex", "gap": "30px", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        style={"flex": "1 1 200px"},
                        children=[
                            html.Label("Lasso α"),
                            dcc.Input(
                                id="lasso-alpha",
                                type="number",
                                value=0.1,
                                step=0.05,
                                style={"width": "100%"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1 1 200px"},
                        children=[
                            html.Label("SVR C"),
                            dcc.Input(
                                id="svr-C",
                                type="number",
                                value=10.0,
                                step=1.0,
                                style={"width": "100%"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1 1 200px"},
                        children=[
                            html.Label("SVR ε"),
                            dcc.Input(
                                id="svr-epsilon",
                                type="number",
                                value=0.1,
                                step=0.01,
                                style={"width": "100%"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1 1 200px"},
                        children=[
                            html.Label("KNN: k neighbors"),
                            dcc.Input(
                                id="knn-k",
                                type="number",
                                value=10,
                                step=1,
                                style={"width": "100%"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1 1 200px"},
                        children=[
                            html.Label("KMeans: # clusters"),
                            dcc.Input(
                                id="kmeans-k",
                                type="number",
                                value=5,
                                step=1,
                                style={"width": "100%"},
                            ),
                        ],
                    ),
                ],
            ),

            html.Br(),
            html.Button("Run Baselines", id="run-btn", n_clicks=0),

            html.Div(id="dataset-info", style={"marginTop": "10px", "fontStyle": "italic"}),

            html.Hr(),

            html.Hr(),
            html.H4("KMeans Unsupervised Diagnostics"),
            html.Pre(id="kmeans-info", style={"whiteSpace": "pre-wrap"}),
        ],
    )


# -----------------------------------------------------------
# Hypergraph Fusion tab layout
# -----------------------------------------------------------
def hypergraph_layout():
    return html.Div(
        style={"maxWidth": "900px"},
        children=[
            html.H3("Hypergraph Fusion Demo (HyperCoCoFusion)"),
            html.P(
                "This tab runs a single forward pass of your HyperCoCoFusion model "
                "on the same simulated_data (fc/sc/morph/cog/labels) to verify that "
                "CMSI, VWV, and spectral hypergraph fusion all compose correctly."
            ),
            html.Label("Data root (same folder as above, with fc.npy/sc.npy/morph.npy/cog.npy/labels.npy)"),
            dcc.Input(
                id="hg-data-root",
                type="text",
                value="simulated_data",
                style={"width": "100%"},
            ),
            html.Br(),
            html.Br(),
            html.Button("Run Hypergraph Forward Pass", id="hg-run-btn", n_clicks=0),
            html.Div(id="hg-metrics", style={"marginTop": "20px"}),
            html.Div(id="hg-preview", style={"marginTop": "10px"}),
        ],
    )


# -----------------------------------------------------------
# Callback: run baselines
# -----------------------------------------------------------
@app.callback(
    Output("metrics-table", "children"),
    Output("r2-figure", "figure"),
    Output("kmeans-info", "children"),
    Output("dataset-info", "children"),
    Input("run-btn", "n_clicks"),
    State("data-root", "value"),
    State("val-frac", "value"),
    State("test-frac", "value"),
    State("seed", "value"),
    State("lasso-alpha", "value"),
    State("svr-C", "value"),
    State("svr-epsilon", "value"),
    State("knn-k", "value"),
    State("kmeans-k", "value"),
    prevent_initial_call=True,
)
def on_run_click(
    n_clicks,
    data_root,
    val_frac,
    test_frac,
    seed,
    lasso_alpha,
    svr_C,
    svr_epsilon,
    knn_k,
    kmeans_k,
):
    try:
        results, X_shape, y_shape = run_all_models(
            data_root=data_root,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=int(seed),
            lasso_alpha=float(lasso_alpha),
            svr_C=float(svr_C),
            svr_epsilon=float(svr_epsilon),
            knn_k=int(knn_k),
            kmeans_k=int(kmeans_k),
        )
    except Exception as e:
        error_msg = f"Error running models: {e}"
        empty_fig = go.Figure()
        return html.Div(error_msg, style={"color": "red"}), empty_fig, "", ""

    # -------------------------------
    # Build metrics DataFrame
    # -------------------------------
    rows = []
    for model_name, res in results.items():
        for split in ["train", "val", "test"]:
            m = res[split]
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "mse": m["mse"],
                    "mae": m["mae"],
                    "r2": m["r2"],
                }
            )

    metrics_df = pd.DataFrame(rows)

    metrics_table = dash_table.DataTable(
        columns=[{"name": c.upper(), "id": c} for c in metrics_df.columns],
        data=metrics_df.to_dict("records"),
        page_size=12,
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": 11, "padding": "4px"},
    )

    # -------------------------------
    # R² bar figure
    # -------------------------------
    model_names = sorted(results.keys())
    splits = ["train", "val", "test"]

    fig = go.Figure()
    for split in splits:
        r2_values = [results[m][split]["r2"] for m in model_names]
        fig.add_bar(name=split, x=model_names, y=r2_values)

    fig.update_layout(
        barmode="group",
        xaxis_title="Model",
        yaxis_title="R²",
        legend_title="Split",
    )

    # -------------------------------
    # KMeans diagnostics
    # -------------------------------
    km_res = results.get("kmeans", {})
    unsup = km_res.get("unsupervised_metrics", {})
    inertia = unsup.get("inertia", None)
    silhouette = unsup.get("silhouette_train", None)
    if inertia is None:
        kmeans_text = "KMeans diagnostics not available."
    else:
        kmeans_text = (
            f"Inertia: {inertia:.4f}\nSilhouette (train): "
            f"{'nan' if silhouette is None else f'{silhouette:.4f}'}"
        )

    # -------------------------------
    # Dataset info
    # -------------------------------
    N, D = X_shape
    if isinstance(y_shape, tuple) and len(y_shape) == 2:
        n_targets = y_shape[1]
    else:
        n_targets = 1
    dataset_info = f"Loaded X shape: {X_shape}, y shape: {y_shape} (N={N}, D={D}, targets={n_targets})"

    return metrics_table, fig, kmeans_text, dataset_info


# -----------------------------------------------------------
# Callback: run Hypergraph Fusion demo
# -----------------------------------------------------------
@app.callback(
    Output("hg-metrics", "children"),
    Output("hg-preview", "children"),
    Input("hg-run-btn", "n_clicks"),
    State("hg-data-root", "value"),
    prevent_initial_call=True,
)
def on_hg_run(n_clicks, data_root):
    try:
        metrics, preview = run_hypergraph_demo(data_root)
    except Exception as e:
        return (
            html.Div(f"Error in HyperCoCoFusion demo: {e}", style={"color": "red"}),
            html.Div(),
        )

    metrics_table = html.Table(
        [
            html.Tr([html.Th("Metric"), html.Th("Value")])
        ]
        + [
            html.Tr([html.Td(k), html.Td(str(v))])
            for k, v in metrics.items()
        ]
    )

    preview_rows = []
    for row in preview:
        preview_rows.append(
            html.Tr(
                [
                    html.Td(row["idx"]),
                    html.Td(json.dumps(row["y_true"])),
                    html.Td(json.dumps(row["y_pred"])),
                ]
            )
        )

    preview_table = html.Table(
        [
            html.Tr(
                [
                    html.Th("idx"),
                    html.Th("y_true"),
                    html.Th("y_pred"),
                ]
            )
        ]
        + preview_rows
    )

    return (
        html.Div(
            [
                html.H4("HyperCoCoFusion Forward-Pass Metrics"),
                metrics_table,
            ]
        ),
        html.Div(
            [
                html.H4("First 5 Predictions vs True"),
                preview_table,
            ]
        ),
    )


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    # For local dev; hosting platforms usually call `server` via Gunicorn.
    app.run(debug=True, host="0.0.0.0", port=8050)
