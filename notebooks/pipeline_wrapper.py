import sys

sys.path.append("src/")
sys.path.append("codif-ape-train/src/")
from camembert.camembert_model import CustomCamembertModel
from camembert.custom_pipeline import CustomPipeline
from utils.mappings import mappings
from transformers import CamembertTokenizer
from transformers.pipelines import pipeline, PIPELINE_REGISTRY


model = CustomCamembertModel.from_pretrained(
    "camembert/camembert-base",
    num_labels=len(mappings.get("APE_NIV5")),
    categorical_features=["liasse_type", "activ_nat_et", "activ_surf_et", "evenement_type", "cj"],
)

PIPELINE_REGISTRY.register_pipeline(
    "text-classification-cat-features",
    pipeline_class=CustomPipeline,
    pt_model=[CustomCamembertModel],
)
tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")
pipe = pipeline("text-classification-cat-features", model=model, tokenizer=tokenizer)

pipe("Vendeur de chaussures", categorical_inputs=[0, 0, 0, 0, 0], k=2)
