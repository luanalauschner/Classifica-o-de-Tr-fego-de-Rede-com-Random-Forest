from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline


TARGET = "_target"
SOURCE_FILE = "_source_file"

LABEL_ORDER = [
    "BENIGN",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Attack Brute Force",
    "Web Attack SQL Injection",
    "Web Attack XSS",
]


@dataclass
class ExperimentConfig:
    output_dir: str = "resultados"
    task: str = "binaria"
    cv_folds: int = 10
    test_size: float = 0.30
    random_state: int = 42
    max_per_class: int | None = 20000
    chunksize: int = 100000
    class_weight: bool = True
    run_rfe: bool = False
    rfe_cv: int = 10
    rfe_step: int = 5
    rfe_max_rows: int | None = 100000


def clean_column_name(name: object) -> str:
    return re.sub(r"\s+", " ", str(name).strip())


def normalize_label(value: object) -> str | float:
    """Corrige diferenças de grafia e caracteres corrompidos nos rótulos."""
    if pd.isna(value):
        return np.nan

    original = str(value).strip()
    text = re.sub(r"[^a-z0-9]+", " ", original.lower()).strip()

    if text == "benign":
        return "BENIGN"
    if "heartbleed" in text:
        return "Heartbleed"
    if "infiltration" in text or "infilteration" in text:
        return "Infiltration"
    if "portscan" in text or ("port" in text and "scan" in text):
        return "PortScan"
    if text == "ddos" or ("distributed" in text and "dos" in text):
        return "DDoS"
    if "goldeneye" in text:
        return "DoS GoldenEye"
    if "hulk" in text:
        return "DoS Hulk"
    if "slowhttptest" in text:
        return "DoS Slowhttptest"
    if "slowloris" in text:
        return "DoS slowloris"
    if "ftp" in text and "patator" in text:
        return "FTP-Patator"
    if "ssh" in text and "patator" in text:
        return "SSH-Patator"
    if text == "bot" or text.startswith("bot "):
        return "Bot"
    if "web" in text and "brute" in text:
        return "Web Attack Brute Force"
    if "web" in text and ("sql" in text or "injection" in text):
        return "Web Attack SQL Injection"
    if "web" in text and "xss" in text:
        return "Web Attack XSS"

    return original


def make_target(
    labels: pd.Series,
    task: str,
) -> pd.Series:
    """
    Cria a variável que o Random Forest deverá prever.

    binaria:
        BENIGN permanece BENIGN;
        todos os ataques são agrupados como ATTACK.

    multiclasse:
        BENIGN e cada tipo de ataque permanecem separados.
    """

    normalized = labels.map(normalize_label)

    if task == "binaria":
        return normalized.map(
            lambda label: (
                "BENIGN"
                if label == "BENIGN"
                else "ATTACK"
            )
        )

    if task == "multiclasse":
        return normalized

    raise ValueError(
        "Tarefa inválida. Use 'binaria' ou 'multiclasse'."
    )


def _iter_csv_chunks(
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    """
    Lê exclusivamente os arquivos CSV localizados na pasta dados.
    """

    data_dir = Path("dados")

    if not data_dir.exists():
        raise FileNotFoundError(
            f"A pasta dados não foi encontrada: {data_dir.resolve()}"
        )

    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"O caminho dados não é uma pasta: {data_dir.resolve()}"
        )

    csv_files = sorted(
        data_dir.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"Nenhum arquivo CSV foi encontrado em: {data_dir.resolve()}"
        )

    for csv_file in csv_files:
        print(f"Lendo {csv_file.name}")

        for chunk in pd.read_csv(
            csv_file,
            chunksize=chunksize,
            low_memory=False,
        ):
            chunk.columns = [
                clean_column_name(column)
                for column in chunk.columns
            ]

            chunk[SOURCE_FILE] = csv_file.name

            yield chunk

def load_sampled_dataset(
    task: str,
    max_per_class: int | None,
    chunksize: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Lê os CSVs da pasta dados e prepara uma amostra para
    classificação binária ou multiclasse.

    Quando max_per_class é informado, mantém no máximo essa
    quantidade de registros de cada classe.
    """

    rng = np.random.default_rng(random_state)

    reservoirs: dict[str, pd.DataFrame] = {}

    all_chunks: list[pd.DataFrame] = []

    for chunk in _iter_csv_chunks(chunksize):

        label_col = next(
            (
                column
                for column in chunk.columns
                if column.lower() == "label"
            ),
            None,
        )

        if label_col is None:
            raise KeyError(
                "A coluna Label não foi encontrada em um dos arquivos CSV."
            )

        # Cria o alvo de acordo com a tarefa escolhida.
        chunk[TARGET] = make_target(
            chunk[label_col],
            task,
        )

        # Remove registros que não possuem rótulo válido.
        chunk = chunk.dropna(
            subset=[TARGET]
        ).copy()

        # Se não houver limite, guarda o bloco completo.
        if max_per_class is None or max_per_class <= 0:
            all_chunks.append(chunk)
            continue

        # Separa os registros por classe.
        for class_name, group in chunk.groupby(
            TARGET,
            sort=False,
        ):
            group = group.copy()

            # Atribui uma prioridade aleatória a cada registro.
            group["_priority"] = rng.random(
                len(group)
            )

            # Recupera os registros já selecionados
            # anteriormente para essa classe.
            previous = reservoirs.get(
                str(class_name)
            )

            if previous is not None:
                group = pd.concat(
                    [previous, group],
                    ignore_index=True,
                )

            # Mantém até max_per_class registros da classe.
            reservoirs[str(class_name)] = group.nsmallest(
                min(max_per_class, len(group)),
                "_priority",
            )

    if max_per_class is None or max_per_class <= 0:

        if not all_chunks:
            raise ValueError(
                "Nenhum registro válido foi encontrado."
            )

        data = pd.concat(
            all_chunks,
            ignore_index=True,
        )

    else:

        if not reservoirs:
            raise ValueError(
                "Nenhum registro válido foi encontrado."
            )

        data = pd.concat(
            reservoirs.values(),
            ignore_index=True,
        )

        data = data.drop(
            columns=["_priority"],
            errors="ignore",
        )

    # Embaralha os registros antes de retorná-los.
    return data.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(
        drop=True
    )

def prepare_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    label_col = next((c for c in data.columns if c.lower() == "label"), None)
    if label_col is None:
        raise KeyError("A coluna Label não foi encontrada.")

    y = data[TARGET].astype(str).copy()
    source = data[SOURCE_FILE].astype(str).copy()

    excluded = {label_col, TARGET, SOURCE_FILE}
    X = data.drop(columns=list(excluded), errors="ignore").copy()

    # A base possui uma segunda cópia da coluna Fwd Header Length.
    duplicate_header_cols = [
        c for c in X.columns
        if c.lower().replace(" ", "") in {"fwdheaderlength.1", "fwdheaderlength1"}
    ]
    X = X.drop(columns=duplicate_header_cols, errors="ignore")

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    constant_cols = [
        col for col in X.columns
        if X[col].nunique(dropna=True) <= 1
    ]
    X = X.drop(columns=constant_cols, errors="ignore")

    return X.astype(np.float32), y, source


def build_model(
    random_state: int,
    use_class_weight: bool,
) -> Pipeline:
    """
    Cria o pipeline de classificação com Random Forest.
    """

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        class_weight=(
            "balanced_subsample"
            if use_class_weight
            else None
        ),
    )

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "model",
                random_forest,
            ),
        ]
    )


def _attack_probability(estimator, X) -> np.ndarray:
    probabilities = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    return probabilities[:, classes.index("ATTACK")]


def roc_auc_attack_scorer(estimator, X, y_true) -> float:
    binary = (np.asarray(y_true) == "ATTACK").astype(int)
    return float(roc_auc_score(binary, _attack_probability(estimator, X)))


def average_precision_attack_scorer(estimator, X, y_true) -> float:
    binary = (np.asarray(y_true) == "ATTACK").astype(int)
    return float(average_precision_score(binary, _attack_probability(estimator, X)))


def scorers(task: str) -> dict[str, object]:
    result = {
        "accuracy": "accuracy",
        "precision_macro": make_scorer(
            precision_score, average="macro", zero_division=0
        ),
        "recall_macro": make_scorer(
            recall_score, average="macro", zero_division=0
        ),
        "f1_macro": make_scorer(
            f1_score, average="macro", zero_division=0
        ),
        "f1_weighted": make_scorer(
            f1_score, average="weighted", zero_division=0
        ),
    }
    if task == "binaria":
        result["roc_auc"] = roc_auc_attack_scorer
        result["average_precision"] = average_precision_attack_scorer
    return result


def save_distribution(y: pd.Series, output: Path) -> None:
    counts = y.value_counts()
    counts.rename_axis("classe").reset_index(name="quantidade").to_csv(
        output.with_suffix(".csv"), index=False
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Distribuição das classes utilizadas")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=170)
    plt.close(fig)


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: Sequence[str],
    title: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=list(labels),
        xticks_rotation=45,
        values_format="d",
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_binary_curves(
    y_true: pd.Series,
    probabilities: np.ndarray,
    output_dir: Path,
    model_name: str,
) -> dict[str, float]:
    binary = (y_true == "ATTACK").astype(int).to_numpy()
    roc_auc = roc_auc_score(binary, probabilities)
    ap = average_precision_score(binary, probabilities)

    fpr, tpr, _ = roc_curve(binary, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title(f"Curva ROC — {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"curva_roc_{model_name}.png", dpi=170)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(binary, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisão")
    ax.set_title(f"Curva Precisão-Recall — {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"curva_pr_{model_name}.png", dpi=170)
    plt.close(fig)

    return {"roc_auc": float(roc_auc), "average_precision": float(ap)}


def save_feature_importance(
    feature_names: Sequence[str],
    importance: np.ndarray,
    output_dir: Path,
    suffix: str = "rf",
) -> pd.DataFrame:
    table = (
        pd.DataFrame({"atributo": list(feature_names), "importancia": importance})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    table.to_csv(output_dir / f"importancia_atributos_{suffix}.csv", index=False)

    top = table.head(20).sort_values("importancia")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["atributo"], top["importancia"])
    ax.set_xlabel("Importância de Gini")
    ax.set_title("20 atributos mais importantes")
    fig.tight_layout()
    fig.savefig(output_dir / f"importancia_atributos_{suffix}.png", dpi=180)
    plt.close(fig)
    return table

def run_rfecv(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    config: ExperimentConfig,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.rfe_max_rows and len(X) > config.rfe_max_rows:
        indices, _ = train_test_split(
            np.arange(len(y)),
            train_size=config.rfe_max_rows,
            stratify=y,
            random_state=config.random_state,
        )
        X_use = X.iloc[indices]
        y_use = y.iloc[indices]
    else:
        X_use = X
        y_use = y

    X_train, X_test, y_train, y_test = train_test_split(
        X_use,
        y_use,
        test_size=config.test_size,
        stratify=y_use,
        random_state=config.random_state,
    )

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)

    estimator = RandomForestClassifier(
        n_estimators=100,
        random_state=config.random_state,
        n_jobs=1,
        class_weight="balanced_subsample" if config.class_weight else None,
    )
    selector = RFECV(
        estimator=estimator,
        step=max(1, config.rfe_step),
        cv=StratifiedKFold(
            n_splits=config.rfe_cv,
            shuffle=True,
            random_state=config.random_state,
        ),
        scoring="f1_macro",
        n_jobs=-1,
        min_features_to_select=min(10, X_train.shape[1]),
    )
    selector.fit(X_train_i, y_train)

    selected = list(X.columns[selector.support_])
    pd.DataFrame({"atributo": selected}).to_csv(
        output_dir / "atributos_selecionados.csv",
        index=False,
    )

    final_model = RandomForestClassifier(
        n_estimators=100,
        random_state=config.random_state,
        n_jobs=-1,
        class_weight="balanced_subsample" if config.class_weight else None,
    )
    final_model.fit(X_train_i[:, selector.support_], y_train)
    pred = final_model.predict(X_test_i[:, selector.support_])

    save_feature_importance(
        selected,
        final_model.feature_importances_,
        output_dir,
        suffix="rfe",
    )

    result = {
        "atributos_originais": int(X.shape[1]),
        "atributos_selecionados": int(selector.n_features_),
        "acuracia_holdout": float(accuracy_score(y_test, pred)),
        "f1_macro_holdout": float(
            f1_score(y_test, pred, average="macro", zero_division=0)
        ),
        "lista_atributos": selected,
    }
    (output_dir / "resumo_rfe.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _write_summary(
    output_dir: Path,
    config: ExperimentConfig,
    X: pd.DataFrame,
    y: pd.Series,
    cv_summary: pd.DataFrame,
    holdout: pd.DataFrame,
    rfe_result: dict[str, object] | None,
) -> None:
    """
    Salva um resumo textual dos resultados do Random Forest.
    """

    lines = [
        f"# Resultados — tarefa {config.task}",
        "",
        f"- Registros utilizados: **{len(y)}**",
        f"- Atributos após limpeza: **{X.shape[1]}**",
        "- Modelo utilizado: **Random Forest**",
        f"- Validação cruzada: **{config.cv_folds} folds estratificados**",
        "",
        "## Métricas de validação cruzada",
        "",
        cv_summary.to_markdown(index=False),
        "",
        "## Métricas do conjunto de teste",
        "",
        holdout.to_markdown(index=False),
        "",
    ]

    if rfe_result is not None:
        lines += [
            "## Seleção recursiva de atributos",
            "",
            (
                "- Atributos originais: "
                f"**{rfe_result['atributos_originais']}**"
            ),
            (
                "- Atributos selecionados: "
                f"**{rfe_result['atributos_selecionados']}**"
            ),
            (
                "- F1 macro no conjunto de teste: "
                f"**{rfe_result['f1_macro_holdout']:.4f}**"
            ),
            "",
        ]

    lines += [
        "## Interpretação",
        "",
    ]

    if config.task == "binaria":
        lines.append(
            "Na tarefa binária, o Random Forest identifica se cada fluxo "
            "pertence à classe BENIGN ou ATTACK."
        )
    else:
        lines.append(
            "Na tarefa multiclasse, o Random Forest identifica se o fluxo "
            "é benigno ou a qual tipo específico de ataque ele pertence."
        )

    lines += [
        "",
        (
            "Os resultados devem ser interpretados considerando o "
            "desbalanceamento entre as classes, a possibilidade de falsos "
            "positivos e falsos negativos e a dependência dos padrões "
            "existentes no CICIDS2017."
        ),
    ]

    (output_dir / "resumo_resultados.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

def run_experiment(
    config: ExperimentConfig,
) -> dict[str, object]:
    """
    Executa o experimento completo utilizando Random Forest.
    """

    output_dir = Path(config.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Salva os parâmetros utilizados na execução.
    (output_dir / "configuracao.json").write_text(
        json.dumps(
            asdict(config),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Lê os arquivos da pasta dados.
    data = load_sampled_dataset(
        config.task,
        config.max_per_class,
        config.chunksize,
        config.random_state,
    )

    # Separa os atributos, os rótulos e o arquivo de origem.
    X, y, source = prepare_xy(data)

    if y.nunique() < 2:
        raise ValueError(
            "A tarefa possui menos de duas classes."
        )

    minimum_class = int(
        y.value_counts().min()
    )

    if config.cv_folds > minimum_class:
        raise ValueError(
            f"cv_folds={config.cv_folds} é maior que "
            f"a menor classe ({minimum_class})."
        )

    # Salva a distribuição das classes.
    save_distribution(
        y,
        output_dir / "distribuicao_classes",
    )

    # Salva a quantidade de registros por arquivo e classe.
    source_table = (
        pd.DataFrame(
            {
                "arquivo": source,
                "classe": y,
            }
        )
        .value_counts()
        .rename("quantidade")
        .reset_index()
    )

    source_table.to_csv(
        output_dir / "distribuicao_por_arquivo.csv",
        index=False,
    )

    # Cria o Random Forest.
    model = build_model(
        config.random_state,
        config.class_weight,
    )

    # Configura a validação cruzada estratificada.
    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    scoring = scorers(
        config.task
    )

    print(
        "Validação cruzada: Random Forest"
    )

    # Executa a validação cruzada.
    cv_result = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
    )

    cv_rows = []

    for fold in range(config.cv_folds):
        row = {
            "modelo": "Random Forest",
            "fold": fold + 1,
            "accuracy": cv_result[
                "test_accuracy"
            ][fold],
            "precision_macro": cv_result[
                "test_precision_macro"
            ][fold],
            "recall_macro": cv_result[
                "test_recall_macro"
            ][fold],
            "f1_macro": cv_result[
                "test_f1_macro"
            ][fold],
            "f1_weighted": cv_result[
                "test_f1_weighted"
            ][fold],
            "fit_time_s": cv_result[
                "fit_time"
            ][fold],
        }

        if "test_roc_auc" in cv_result:
            row["roc_auc"] = cv_result[
                "test_roc_auc"
            ][fold]

        if "test_average_precision" in cv_result:
            row["average_precision"] = cv_result[
                "test_average_precision"
            ][fold]

        cv_rows.append(row)

    # Salva as métricas de cada fold.
    cv_details = pd.DataFrame(
        cv_rows
    )

    cv_details.to_csv(
        output_dir / "metricas_cv_por_fold.csv",
        index=False,
    )

    # Calcula as médias da validação cruzada.
    cv_summary_row = {
        "modelo": "Random Forest",
        "acuracia_media": float(
            cv_details["accuracy"].mean()
        ),
        "acuracia_desvio": float(
            cv_details["accuracy"].std()
        ),
        "precisao_macro_media": float(
            cv_details["precision_macro"].mean()
        ),
        "recall_macro_medio": float(
            cv_details["recall_macro"].mean()
        ),
        "f1_macro_medio": float(
            cv_details["f1_macro"].mean()
        ),
        "f1_ponderado_medio": float(
            cv_details["f1_weighted"].mean()
        ),
        "tempo_treino_medio_s": float(
            cv_details["fit_time_s"].mean()
        ),
    }

    if "roc_auc" in cv_details.columns:
        cv_summary_row["roc_auc_medio"] = float(
            cv_details["roc_auc"].mean()
        )

    if "average_precision" in cv_details.columns:
        cv_summary_row[
            "average_precision_media"
        ] = float(
            cv_details[
                "average_precision"
            ].mean()
        )

    cv_summary = pd.DataFrame(
        [cv_summary_row]
    )

    cv_summary.to_csv(
        output_dir / "resumo_metricas_cv.csv",
        index=False,
    )

    # Gera o gráfico das acurácias obtidas nos folds.
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.boxplot(
        [cv_details["accuracy"].to_numpy()],
        tick_labels=["Random Forest"],
    )

    ax.set_title(
        "Acurácia do Random Forest nos folds"
    )

    ax.set_xlabel(
        "Modelo"
    )

    ax.set_ylabel(
        "Acurácia"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "boxplot_acuracia_cv.png",
        dpi=170,
    )

    plt.close(fig)

    # Separa os registros para treinamento e teste final.
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )

    X_train = X.iloc[
        train_idx
    ]

    X_test = X.iloc[
        test_idx
    ]

    y_train = y.iloc[
        train_idx
    ]

    y_test = y.iloc[
        test_idx
    ]

    print(
        "Treinamento final: Random Forest"
    )

    # Cria e treina um novo Random Forest com os dados de treino.
    fitted_model = build_model(
        config.random_state,
        config.class_weight,
    )

    fitted_model.fit(
        X_train,
        y_train,
    )

    # Realiza as previsões no conjunto de teste.
    predictions = fitted_model.predict(
        X_test
    )

    holdout_row = {
        "modelo": "Random Forest",
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "precision_macro": float(
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    # Gera as curvas ROC e Precisão-Recall na tarefa binária.
    if (
        config.task == "binaria"
        and hasattr(
            fitted_model,
            "predict_proba",
        )
    ):
        class_index = list(
            fitted_model.classes_
        ).index(
            "ATTACK"
        )

        probabilities = fitted_model.predict_proba(
            X_test
        )[:, class_index]

        holdout_row.update(
            save_binary_curves(
                y_test,
                probabilities,
                output_dir,
                "RF",
            )
        )

    holdout = pd.DataFrame(
        [holdout_row]
    )

    holdout.to_csv(
        output_dir / "metricas_holdout.csv",
        index=False,
    )

    # Salva o relatório de classificação.
    labels = sorted(
        y.unique()
    )

    report = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
    ).T

    report.to_csv(
        output_dir
        / "relatorio_classificacao_RF.csv"
    )

    # Salva a matriz de confusão.
    save_confusion_matrix(
        y_test,
        predictions,
        labels,
        "Matriz de confusão — Random Forest",
        output_dir / "matriz_confusao_RF.png",
    )

    # Salva a importância dos atributos.
    random_forest = fitted_model.named_steps[
        "model"
    ]

    save_feature_importance(
        X.columns,
        random_forest.feature_importances_,
        output_dir,
        suffix="rf",
    )

    # Salva o modelo treinado.
    artifact = {
        "model": fitted_model,
        "feature_names": list(
            X.columns
        ),
        "task": config.task,
        "labels": labels,
        "model_name": "Random Forest",
    }

    joblib.dump(
        artifact,
        output_dir
        / "modelo_random_forest.joblib",
    )

    # Executa a seleção de atributos, quando solicitada.
    rfe_result = None

    if config.run_rfe:
        print(
            "Executando RFECV. Esta etapa pode demorar."
        )

        rfe_result = run_rfecv(
            X,
            y,
            output_dir / "rfe",
            config,
        )

    # Gera o resumo em Markdown.
    _write_summary(
        output_dir,
        config,
        X,
        y,
        cv_summary,
        holdout,
        rfe_result,
    )

    # Resultado apresentado no terminal.
    result = {
        "task": config.task,
        "rows": int(
            len(y)
        ),
        "features": int(
            X.shape[1]
        ),
        "model": "Random Forest",
        "cv_f1_macro": float(
            cv_summary.iloc[0][
                "f1_macro_medio"
            ]
        ),
        "cv_accuracy": float(
            cv_summary.iloc[0][
                "acuracia_media"
            ]
        ),
        "output_dir": str(
            output_dir
        ),
    }

    (output_dir / "resumo_execucao.json").write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    return result

def predict_saved_model(
    model_path: str | Path,
    csv_path: str | Path,
    output_path: str | Path,
) -> Path:
    artifact = joblib.load(model_path)
    data = pd.read_csv(csv_path, low_memory=False)
    data.columns = [clean_column_name(c) for c in data.columns]

    # O alvo é apenas um campo auxiliar para reutilizar a limpeza.
    data[TARGET] = "UNKNOWN"
    data[SOURCE_FILE] = Path(csv_path).name
    X, _, _ = prepare_xy(data)
    X = X.reindex(columns=artifact["feature_names"])

    pred = artifact["model"].predict(X)
    result = pd.DataFrame({"predicao": pred})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return output_path
