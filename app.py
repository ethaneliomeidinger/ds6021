import json
import os

# Make DGL skip GraphBolt to avoid extra deps issues
os.environ["DGL_GRAPHBOLT_LOAD"] = "0"

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go

import benchmark as bm

# PyTorch + DGL-related imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from encoders import GraphSAGEEncoder, DummyEncoder
from hyperfusion import HyperCoCoFusion
from datasets import MultimodalDGLDataset, multimodal_dgl_collate_fn

import random


# -----------------------------------------------------------
# Wrapper utilities for encoders + normalization
# -----------------------------------------------------------

class GraphSageWrapper(nn.Module):
    """
    Simple wrapper around your GraphSAGEEncoder that expects (g) only.
    Not strictly needed in the current wiring, but kept for flexibility.
    """
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
        x = g.ndata["feat"]
        return self.gnn(g, x)


class PullX(nn.Module):
    """Wrap a (g, x) graph encoder so it works with just (g)."""
    def __init__(self, base_encoder: nn.Module):
        super().__init__()
        self.base = base_encoder

    def forward(self, g):
        # prefer 'x'; fall back to 'feat'
        if "x" in g.ndata:
            x = g.ndata["x"]
        elif "feat" in g.ndata:
            x = g.ndata["feat"]
            g.ndata["x"] = x
        else:
            raise KeyError("Graph is missing node features: need g.ndata['x'] or ['feat'].")
        return self.base(g, x)


class FeatureAdapter(nn.Module):
    """Wrap a feature-only encoder so it accepts a single tensor argument."""
    def __init__(self, feat_encoder: nn.Module):
        super().__init__()
        self.feat_encoder = feat_encoder

    def forward(self, x):
        return self.feat_encoder(x)


def normalize_graph_inplace(g, feat_key_src="feat", feat_key_dst="x"):
    """
    Copy node feats from feat_key_src -> feat_key_dst if needed, then
    apply signed log1p + z-score + clipping (like your training script).
    """
    if feat_key_dst not in g.ndata:
        if feat_key_src in g.ndata:
            g.ndata[feat_key_dst] = g.ndata[feat_key_src]
        else:
            raise KeyError("Graph is missing node feats: need ndata['x'] or ndata['feat'].")

    x = g.ndata[feat_key_dst].float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # signed log1p
    x = x.sign() * torch.log1p(x.abs())

    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std
    x = x.clamp_(-8.0, 8.0)

    g.ndata[feat_key_dst] = x
    return g


def _build_encoder(encoder_name: str, D: int, cmsi_dim: int, morph_in: int):
    """
    Build encoders similar to your training script:
      - graph encoder: GAT / GraphSAGE / GIN (you’ll use 'gsage' in this app)
      - morph encoder: DummyEncoder over (B, R, d)
    """
    import encoders as encoders1  # to match your script’s namespace

    if encoder_name == "gin":
        enc_fc_base = encoders1.GINEncoder(in_feats=D, out_dim=cmsi_dim)
        enc_sc_base = encoders1.GINEncoder(in_feats=D, out_dim=cmsi_dim)
    elif encoder_name == "gsage":
        enc_fc_base = encoders1.GraphSAGEEncoder(in_feats=D, out_dim=cmsi_dim)
        enc_sc_base = encoders1.GraphSAGEEncoder(in_feats=D, out_dim=cmsi_dim)
    else:  # "gat"
        enc_fc_base = encoders1.GATEncoder(in_feats=D, out_dim=cmsi_dim)
        enc_sc_base = encoders1.GATEncoder(in_feats=D, out_dim=cmsi_dim)

    # morph_in is the last dim of morph: (B, R, d) → d
    enc_morph_base = encoders1.DummyEncoder(in_dim=morph_in, out_dim=cmsi_dim)

    encoder_fc = PullX(enc_fc_base)
    encoder_sc = PullX(enc_sc_base)
    encoder_morph = FeatureAdapter(enc_morph_base)

    return encoder_fc, encoder_sc, encoder_morph


# -----------------------------------------------------------
# Baseline models: Lasso, SVR, KNN, KMeans
# -----------------------------------------------------------

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

    # 1) Load multi-modal concatenated features
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
# Hypergraph (HyperCoCoFusion) training helpers
# -----------------------------------------------------------

def _split_indices(n, seed=42, train_ratio=0.7, val_ratio=0.15):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


@torch.no_grad()
def _evaluate(model, dl, device):
    model.eval()
    crit = torch.nn.MSELoss()
    total, count = 0.0, 0
    for g_fc, g_sc, morph, cog, labels in dl:
        normalize_graph_inplace(g_fc)
        normalize_graph_inplace(g_sc)
        morph, cog, labels = morph.to(device), cog.to(device), labels.to(device)
        out = model(g_fc, g_sc, morph, cog)
        loss = crit(out, labels)
        if torch.isfinite(loss):
            total += loss.item()
            count += 1
    return float("nan") if count == 0 else total / count


def _train_hypergraph(model, train_dl, val_dl, device, epochs=15, lr=1e-4):
    crit = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_hist, val_hist = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running, batches = 0.0, 0

        for g_fc, g_sc, morph, cog, labels in train_dl:
            normalize_graph_inplace(g_fc)
            normalize_graph_inplace(g_sc)
            morph, cog, labels = morph.to(device), cog.to(device), labels.to(device)

            opt.zero_grad(set_to_none=True)
            out = model(g_fc, g_sc, morph, cog)
            loss = crit(out, labels)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            batches += 1

        train_loss = float("nan") if batches == 0 else running / batches
        val_loss = _evaluate(model, val_dl, device)

        train_hist.append(train_loss)
        val_hist.append(val_loss)

    return train_hist, val_hist


# -----------------------------------------------------------
# Dash app setup
# -----------------------------------------------------------

app = Dash(__name__)
server = app.server  # for Gunicorn / Render
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
                dcc.Tab(label="Data Description", value="tab-data-description"),
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
    elif tab == "tab-data-description":
        return data_description_layout()
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
                    html.Li("K-Means cluster-mean regression"),
                    html.Li("HyperCoCoFusion (spectral hypergraph + CMSI, demo tab)"),
                ]
            ),
        ],
    )


# -----------------------------------------------------------
# Data Description tab
# -----------------------------------------------------------

def data_description_layout():
    return html.Div(
        style={"maxWidth": "900px", "lineHeight": "1.6"},
        children=[
            html.H2("Data Description"),

            html.H3("Neuroimaging Modalities"),
            html.Img(
                src="/assets/fmri.png",
                style={
                    "width": "100%",
                    "border": "1px solid #ccc",
                    "borderRadius": "6px",
                    "marginBottom": "20px",
                },
            ),

            html.H4("Functional Connectivity (FC)"),
            html.P(
                "Functional connectivity is derived from correlations in resting-state fMRI "
                "time series across cortical regions. FC captures distributed patterns of "
                "co-activation and reflects large-scale functional organization, including "
                "default mode, attention, and sensorimotor systems."
            ),

            html.H4("Structural Connectivity (SC)"),
            html.P(
                "Structural connectivity is computed from diffusion MRI using tractography, "
                "estimating the white-matter pathways linking cortical regions. SC reflects "
                "anatomical communication routes supporting information flow across the brain."
            ),

            html.H4("Morphology (Cortical Morphometry)"),
            html.P(
                "Morphological features include cortical thickness, surface area, curvature, "
                "and other geometry-derived measurements across regions of the cortex. These "
                "features index structural maturation, cortical expansion, and neurodevelopmental "
                "variation relevant to cognitive and behavioral outcomes."
            ),

            html.Hr(),

            html.H3("Behavioral Labels (CBCL Scores)"),
            html.P(
                "The Child Behavior Checklist (CBCL/6–18; Achenbach & Rescorla, 2001) provides "
                "standardized parent-reported assessments of emotional and behavioral functioning "
                "in youth (Chavannes & Gignac, 2024). We use three composite age- and sex-"
                "normalized T-scores:"
            ),

            html.H4("Total Problems"),
            html.P(
                "A global index of behavioral and emotional dysregulation. It aggregates all "
                "syndrome scales, including attention problems, anxiety, depressive symptoms, "
                "social difficulties, rule-breaking behavior, and aggression."
            ),

            html.H4("Internalizing Problems"),
            html.P(
                "Reflects inwardly directed distress, combining Anxious/Depressed, "
                "Withdrawn/Depressed, and Somatic Complaints subscales. Elevated scores "
                "indicate anxiety, social withdrawal, depressive affect, and somatic concerns."
            ),

            html.H4("Externalizing Problems"),
            html.P(
                "Measures outwardly directed dysregulation, including Rule-Breaking Behavior "
                "and Aggressive Behavior subscales. High scores indicate impulsivity, "
                "behavioral disinhibition, and difficulty regulating conflict or anger."
            ),

            html.Hr(),

            html.H3("Cognitive Assessments"),
            html.P("The following NIH Toolbox tasks are included:"),
            html.Ul(
                [
                    html.Li("Dimensional Card Change Sort"),
                    html.Li("Flanker Inhibitory Control and Attention"),
                    html.Li("List Sort Working Memory"),
                    html.Li("Oral Reading Recognition"),
                    html.Li("Pattern Comparison Processing Speed"),
                    html.Li("Picture Sequence Memory"),
                    html.Li("Picture Vocabulary"),
                ]
            ),
            html.P(
                "Together, these tasks probe executive control, working memory, processing speed, "
                "language, and episodic memory, providing a cognitive context for the brain measures."
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
                            html.Label(
                                "Data root (folder with cog.npy, morph.npy, sc.npy, fc.npy, labels.npy)"
                            ),
                            dcc.Input(
                                id="data-root",
                                type="text",
                                value="simulated_data",
                                style={"width": "100%"},
                            ),
                            html.Small(
                                "Update this path if your .npy files live somewhere else.",
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
                                marks={
                                    float(v): f"{int(v * 100)}%"
                                    for v in np.arange(0.05, 0.35, 0.05)
                                },
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
                                marks={
                                    float(v): f"{int(v * 100)}%"
                                    for v in np.arange(0.05, 0.35, 0.05)
                                },
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

            html.Div(
                id="dataset-info",
                style={"marginTop": "10px", "fontStyle": "italic"},
            ),

            html.Hr(),
            html.H4("Model Performance (MSE, MAE, R²)"),
            html.Div(id="metrics-table"),

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
                "This tab trains HyperCoCoFusion for a small number of epochs "
                "on the simulated multimodal dataset (fc/sc/morph/cog/labels). "
                "We use GraphSAGE encoders for FC and SC, and a DummyEncoder for morphology."
            ),

            html.Label(
                "Data root (folder with fc.npy/sc.npy/morph.npy/cog.npy/labels.npy)"
            ),
            dcc.Input(
                id="hyper-data-root",
                type="text",
                value="simulated_data",
                style={"width": "100%"},
            ),
            html.Br(), html.Br(),

            html.Label("Training epochs"),
            dcc.Slider(
                id="hyper-epochs",
                min=1,
                max=30,
                step=1,
                value=15,
                marks={1: "1", 5: "5", 15: "15", 30: "30"},
            ),
            html.Br(),

            html.Button(
                "Train Hypergraph Model", id="hyper-train-btn", n_clicks=0
            ),

            html.Div(
                id="hyper-status",
                style={
                    "whiteSpace": "pre-wrap",
                    "fontFamily": "monospace",
                    "marginTop": "12px",
                    "color": "#333",
                },
            ),

            html.Hr(),
            html.H4("Training vs Validation Loss (MSE)"),
            dcc.Graph(id="hyper-loss-fig"),
        ],
    )


# -----------------------------------------------------------
# Callback: run baselines
# -----------------------------------------------------------

@app.callback(
    Output("metrics-table", "children"),
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
        return (
            html.Div(error_msg, style={"color": "red"}),
            "",
            "",
        )

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
    dataset_info = (
        f"Loaded X shape: {X_shape}, y shape: {y_shape} "
        f"(N={N}, D={D}, targets={n_targets})"
    )

    return metrics_table, kmeans_text, dataset_info


# -----------------------------------------------------------
# Callback: train Hypergraph Fusion model
# -----------------------------------------------------------

@app.callback(
    Output("hyper-status", "children"),
    Output("hyper-loss-fig", "figure"),
    Input("hyper-train-btn", "n_clicks"),
    State("hyper-data-root", "value"),
    State("hyper-epochs", "value"),
    prevent_initial_call=True,
)
def train_hypergraph_callback(n_clicks, data_root, epochs):
    try:
        device = torch.device("cpu")

        # 1) Dataset + splits
        full_ds = MultimodalDGLDataset(
            root_dir=data_root, threshold=0.0, mode="topk", k=30
        )
        N = len(full_ds)
        train_idx, val_idx, test_idx = _split_indices(
            N, seed=301, train_ratio=0.7, val_ratio=0.15
        )

        train_ds = torch.utils.data.Subset(full_ds, train_idx)
        val_ds = torch.utils.data.Subset(full_ds, val_idx)
        test_ds = torch.utils.data.Subset(full_ds, test_idx)

        train_dl = DataLoader(
            train_ds,
            batch_size=32,
            shuffle=True,
            collate_fn=multimodal_dgl_collate_fn,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=32,
            shuffle=False,
            collate_fn=multimodal_dgl_collate_fn,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=32,
            shuffle=False,
            collate_fn=multimodal_dgl_collate_fn,
        )

        # 2) Peek at one batch to infer dims
        g_fc_b, g_sc_b, morph_b, cog_b, labels_b = next(iter(train_dl))
        if "x" not in g_fc_b.ndata and "feat" in g_fc_b.ndata:
            g_fc_b.ndata["x"] = g_fc_b.ndata["feat"]
        if "x" not in g_sc_b.ndata and "feat" in g_sc_b.ndata:
            g_sc_b.ndata["x"] = g_sc_b.ndata["feat"]

        D = g_fc_b.ndata["x"].shape[-1]  # node feature dim
        morph_in = morph_b.shape[-1]     # per-ROI morph dim (last dim)
        vwv_d = cog_b.shape[-1]          # cognitive dim
        C = labels_b.shape[-1]           # labels
        cmsi_dim = 32

        # 3) Build encoders and model (GraphSAGE version)
        encoder_fc, encoder_sc, encoder_morph = _build_encoder(
            encoder_name="gsage",
            D=D,
            cmsi_dim=cmsi_dim,
            morph_in=morph_in,
        )
        encoder_fc.to(device)
        encoder_sc.to(device)
        encoder_morph.to(device)

        model = HyperCoCoFusion(
            encoder_fc=encoder_fc,
            encoder_sc=encoder_sc,
            encoder_morph=encoder_morph,
            cmsi_d=cmsi_dim,
            vwv_d=vwv_d,
            label_dim=C,
            cmsi_method="cross_attention",
            info_theory=False,
            vwv_riemann=True,
            use_spectral_hypergraph=True,
            use_residual=False,
        ).to(device)

        # 4) Train
        train_hist, val_hist = _train_hypergraph(
            model, train_dl, val_dl, device, epochs=int(epochs), lr=1e-4
        )

        # 5) Final test loss
        test_loss = _evaluate(model, test_dl, device)

        status = (
            f"Trained HyperCoCoFusion for {epochs} epochs on {data_root}.\n"
            f"Train loss (last epoch): {train_hist[-1]:.4f}\n"
            f"Val   loss (last epoch): {val_hist[-1]:.4f}\n"
            f"Test loss              : {test_loss:.4f}"
        )

        # 6) Build loss curve
        fig = go.Figure()
        fig.add_scatter(
            x=list(range(1, len(train_hist) + 1)),
            y=train_hist,
            mode="lines+markers",
            name="train",
        )
        fig.add_scatter(
            x=list(range(1, len(val_hist) + 1)),
            y=val_hist,
            mode="lines+markers",
            name="val",
        )
        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="MSE loss",
            legend_title="Split",
        )

        return status, fig

    except Exception as e:
        err_fig = go.Figure()
        return f"Error while training HyperCoCoFusion: {e}", err_fig


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
