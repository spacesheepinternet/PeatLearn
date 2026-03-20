#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from peatlearn.adaptive.topic_model import CorpusTopicModel

def main():
    tm = CorpusTopicModel(
        corpus_glob=str(ROOT / "data" / "processed" / "**" / "*.txt"),
        model_dir=str(ROOT / "data" / "models" / "topics"),
        max_features=50000,
        min_df=2,
        max_df=0.6,
        n_components_svd=300,
    )
    tm.build()
    print("Topics built. Artifacts written to data/models/topics and data/reports/topic_clusters_report.md")

if __name__ == "__main__":
    main()
