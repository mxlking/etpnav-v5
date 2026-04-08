from navmorph_compat import patch_habitat_compat

# Apply compatibility patch before importing project extensions.
patch_habitat_compat()

from habitat_extensions import measures, obs_transformers, sensors, nav
from habitat_extensions.config.default import get_extended_config
from habitat_extensions.task import VLNCEDatasetV1
from habitat_extensions.habitat_simulator import Simulator
