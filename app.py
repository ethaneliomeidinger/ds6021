import json

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go

import benchmark as bm  


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

      - load_concatenated_features
      - make_splits
      - run_lasso
      - run_svr
      - run_knn
      - run_kmeans_regression

    Returns:
      results: dict[model_name -> result_dict]
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



# Dash app

app = Dash(__name__)
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
                                value="./sim_data",  # <--- change this to your simulated data path
                                style={"width": "100%"},
                            ),
                            html.Small(
                                "Update this path to point to your simulated HCP-D features.",
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
                                marks={v: f"{int(v*100)}%" for v in np.arange(0.05, 0.35, 0.05)},
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
                                marks={v: f"{int(v*100)}%" for v in np.arange(0.05, 0.35, 0.05)},
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

            html.Div(
                style={"display": "flex", "gap": "40px", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        style={"flex": "1 1 400px"},
                        children=[
                            html.H4("Performance Metrics (MSE, MAE, R²)"),
                            html.Div(id="metrics-table"),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1 1 400px"},
                        children=[
                            html.H4("R² by Model and Split"),
                            dcc.Graph(id="r2-figure"),
                        ],
                    ),
                ],
            ),

            html.Hr(),
            html.H4("KMeans Unsupervised Diagnostics"),
            html.Pre(id="kmeans-info", style={"whiteSpace": "pre-wrap"}),
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
        # In case the path is wrong or files missing, show error
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
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    # When you’re ready to deploy, set debug=False
    app.run_server(debug=True)
