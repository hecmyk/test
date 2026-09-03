"""sys.path pour les notebooks de ce dossier : eval/mmlongbench/, eval/, racine repo.

Les notebooks vivent a cote des modules du banc, mais importent aussi la couche
partagee (analyze, metrics, judges) et le code racine (config, dom_*). Jupyter ne
met que le dossier du notebook sur sys.path : ce shim ajoute les deux autres.
Usage : `import _path` en tete de premiere cellule.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
