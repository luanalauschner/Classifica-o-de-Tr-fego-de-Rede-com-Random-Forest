from __future__ import annotations

import argparse

from cicids2017_ml import predict_saved_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aplica um modelo treinado a um novo CSV."
    )
    parser.add_argument("--modelo", required=True)
    parser.add_argument("--dados", required=True)
    parser.add_argument("--saida", default="predicoes.csv")
    args = parser.parse_args()

    output = predict_saved_model(
        args.modelo,
        args.dados,
        args.saida,
    )
    print(f"Predições salvas em: {output}")


if __name__ == "__main__":
    main()
