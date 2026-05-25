#!/bin/bash
# E.6 smoke test: verify MeltingPot install on DRAC Tamia.
set -euo pipefail

module load StdEnv/2023 python/3.11.5

export VIRTUAL_ENV=/scratch/r/rram17/maps/venv-marl
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

python --version

echo "--- import dmlab2d ---"
python -c "import dmlab2d; print(dmlab2d.__file__)"

echo "--- import meltingpot ---"
python -c "import meltingpot; print(meltingpot.__file__)"

echo "--- substrate build ---"
python <<'PYEOF'
from meltingpot import substrate
cfg = substrate.get_config('commons_harvest__closed')
print('roles:', cfg.default_player_roles[:3])
env = substrate.build('commons_harvest__closed', roles=cfg.default_player_roles)
ts = env.reset()
print('reset OK, num agents:', len(ts.observation))
env.close()
print('=== E.6 SUCCESS ===')
PYEOF
