from transformers import AutoConfig

model_name = "AIPI540/fine_tuned_ViT_for_artwork"

config = AutoConfig.from_pretrained(model_name)

config.save_pretrained('../data/config')
