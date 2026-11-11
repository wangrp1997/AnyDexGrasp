"""Six-object protocol validation around the unchanged external scorer.

The generic scorer is reused with an explicit protocol schema extension in this
process. Its scoring, model loading, geometry workers and outputs are unchanged.
"""
import sys
import json
from pathlib import Path
import evaluate_boya_external as scorer
from prepare_boya_fast6_runtime import validate_fast6_protocol


def main():
    path = Path(sys.argv[sys.argv.index('--protocol') + 1])
    validate_fast6_protocol(json.loads(path.read_text()))
    old = scorer.PROTOCOL_SCHEMA
    try:
        scorer.PROTOCOL_SCHEMA = 'anydex_boya_fast6_protocol_v1'
        scorer.main()
    finally:
        scorer.PROTOCOL_SCHEMA = old


if __name__ == '__main__':
    main()
