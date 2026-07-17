from __future__ import annotations

import argparse

from cicids2017_ml import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimento de classificação com o CICIDS2017."
    )
    parser.add_argument("--saida", default="resultados")
    parser.add_argument( 
        "--tarefa", 
        default="binaria", 
        choices=["binaria", "multiclasse"], 
        help=(
            "binaria: classifica BENIGN ou ATTACK; "
            "multiclasse: classifica BENIGN ou o tipo específico de ataque."
        ),
    )
    parser.add_argument("--cv", type=int, default=10) # validação cruzada com 10 folds
    parser.add_argument("--teste", type=float, default=0.30)
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument(
        "--max-por-classe",
        type=int,
        default=20000,
        help="Use 0 para tentar carregar toda a base.",
    )
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument(
        "--sem-peso-classes",
        action="store_true",
        help="Desativa class_weight nos modelos de árvore.",
    )
    parser.add_argument("--rfe", action="store_true")
    parser.add_argument("--rfe-cv", type=int, default=10)
    parser.add_argument("--rfe-step", type=int, default=5)
    parser.add_argument("--rfe-max-linhas", type=int, default=100000)
    args = parser.parse_args()

    config = ExperimentConfig(
        output_dir=args.saida,
        task=args.tarefa,
        cv_folds=args.cv,
        test_size=args.teste,
        random_state=args.semente,
        max_per_class=None if args.max_por_classe <= 0 else args.max_por_classe,
        chunksize=args.chunksize,
        class_weight=not args.sem_peso_classes,
        run_rfe=args.rfe,
        rfe_cv=args.rfe_cv,
        rfe_step=args.rfe_step,
        rfe_max_rows=args.rfe_max_linhas,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
