from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE
from .dynamic_mean_vfe import DynamicMeanVFE
from .dynamic_pillar_vfe import DynamicPillarVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate
from .pvt_dgcnn import PVTDGCNN
from .pvt_sgcnn_two_stage import Pvt_Sgcnn_Two_Stage
from .pvt_sgcnn_waymo import PVTSGCNN


__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'DynMeanVFE': DynamicMeanVFE,
    'DynPillarVFE': DynamicPillarVFE,
    'PVTDGCNN': PVTDGCNN,
    'Pvt_Sgcnn_Two_Stage': Pvt_Sgcnn_Two_Stage,
    'PVTSGCNN': PVTSGCNN,
}
