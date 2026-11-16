#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -u infer_unetpp_images.py --fold 4 --bbox-mode internal --limit 20 --debug --out-dir infer_outputs/images_unetpp_smoke_fold4 2>&1 | tee infer_unetpp_images_smoke.log
# After verifying smoke output, run the full experiment manually:
# python -u infer_unetpp_images.py --fold 4 --bbox-mode internal --limit 0 --out-dir infer_outputs/images_unetpp_internal_fold4 2>&1 | tee infer_unetpp_images_internal_fold4.log
