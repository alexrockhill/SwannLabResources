from .slowfast import (preproc_slowfast, slowfast_group,
                       slowfast_group_stats, comparison_stats,
                       slowfast2epochs_indices)
from .preprocessing import (get_raw, get_info, get_bads, set_bads,
						    find_ica, apply_ica, get_ica, set_ica,
                            get_aux_epochs, get_ica_components,
                            set_ica_components, mark_autoreject)