#!/bin/sh
# Offline, VERSION-PINNED dependency install for the Arduino UNO Q.
#
# Pinned deliberately: onnxruntime is not compared bit-for-bit across minor
# versions before shipping, so "whatever pip finds" is not good enough for a
# competition submission. --force-reinstall --no-deps means an unpinned
# onnxruntime already on the system (from an earlier `pip install
# onnxruntime` before this script existed) gets overwritten with the exact
# version this model was verified against, instead of pip silently treating
# ANY version as satisfying the requirement.
#
# numpy and scipy are NOT installed here. They come from apt (numpy 2.2.4,
# scipy 1.15.3) and must stay that way - a pip --user copy takes priority on
# sys.path for every script this user runs, board-wide, not just this app.
set -e
cd "$(dirname "$0")"

ORT_VERSION="1.28.0"
echo "installing onnxruntime==${ORT_VERSION} from ./wheels (offline, pinned, no deps) ..."
pip3 install --break-system-packages --no-index --find-links wheels \
    --force-reinstall --no-deps "onnxruntime==${ORT_VERSION}"

echo "installing onnxruntime's small pure-python deps (protobuf, flatbuffers, packaging) ..."
pip3 install --break-system-packages --no-index --find-links wheels \
    protobuf flatbuffers packaging

echo
echo "--- versions actually active now ---"
python3 -c "
import onnxruntime, numpy, scipy
print('onnxruntime', onnxruntime.__version__,
      '(expect exactly ${ORT_VERSION})')
print('numpy      ', numpy.__version__, '(expect the apt version, 2.2.4)')
print('scipy      ', scipy.__version__, '(expect the apt version, 1.15.3)')
print('providers  ', onnxruntime.get_available_providers())
"

python3 - <<'PYEOF'
import numpy, scipy, sys
loc = lambda m: m.__file__
warn = []
if "dist-packages" not in loc(numpy):
    warn.append(f"numpy loaded from {loc(numpy)} - not the apt copy")
if "dist-packages" not in loc(scipy):
    warn.append(f"scipy loaded from {loc(scipy)} - not the apt copy")
if warn:
    print()
    print("WARNING - a pip --user numpy/scipy is shadowing apt:")
    for w in warn:
        print("  " + w)
    print("Fix with:  pip3 uninstall -y numpy scipy")
    print("(this removes the --user copies; apt's versions become active")
    print(" again automatically, nothing to reinstall)")
    sys.exit(1)
PYEOF
