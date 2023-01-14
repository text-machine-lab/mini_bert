import torch
import json
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoModel, AutoConfig
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaEncoder, RobertaLMHead
from typing import List, Optional, Tuple, Union

class LanModelConfig(PretrainedConfig):
    model_type = 'roberta'
    def __init__(
        self,
        **kwargs,
    ):
        default_config = {
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 3,
            "classifier_dropout": 0,
            "eos_token_id": 4,
            "gradient_checkpointing": False,
            "embedding_size": 256,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "layer_norm_eps": 0.00001,
            "max_position_embeddings": 130,
            "model_type": "roberta",
            "num_attention_heads": 8,
            "num_hidden_layers": 8,
            "pad_token_id": 1,
            "position_embedding_type": "absolute",
            "torch_dtype": "float32",
            "transformers_version": "4.24.0",
            "type_vocab_size": 2,
            "use_cache": True,
            "vocab_size": 19000,
        }
        
        #
        for key in kwargs:
            default_config[key] = kwargs[key]
        
        #
        super().__init__(**default_config)
        
        # set attributes
        for key in default_config:
            setattr(self, key, default_config[key])
        
        return

class LanModel(PreTrainedModel):
    config_class = LanModelConfig
    
    def __init__(
        self, 
        config
    ):
        #
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        
        
        
        # start with embedding layer
        config.hidden_size = self.embedding_size
        self.embeddings = RobertaEmbeddings(config)
        
        # add a linear layer for transforming embedding_size into hidden_size
        config.hidden_size = self.hidden_size
        self.emb2hidden = nn.Sequential(
            nn.Linear(
                in_features=config.embedding_size,
                out_features=config.hidden_size
            ),
            nn.LayerNorm(
                config.hidden_size, 
                eps=config.layer_norm_eps
            )
        )
        
        # add transformer model
        self.encoder = RobertaEncoder(config)        
        
        #
        self.head = RobertaLMHead(config)
        
    def forward(self, batch):
        hidden = self.emb2hidden(self.embeddings(batch))
        outputs = self.encoder(hidden)
        outputs.logits = self.head(outputs.last_hidden_state)
        
        return outputs

class LanModelSequenceClassification(PreTrainedModel):
    config_class = LanModelConfig
    
    def __init__(
        self, 
        config
    ):
        #
        super().__init__(config)
        self.config = config        
        
        
        # start with embedding layer
        self.model = config.model
        
        #
        self.pooler = nn.Linear(
            in_features=128,
            out_features=1,
        )
        
        # add a linear layer for transforming embedding_size into hidden_size
        self.head_sequence_classification = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.num_labels,
        )
        
        #
        self.criterion = nn.CrossEntropyLoss()
        
        return
        
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        #print(kwargs.keys())
        #print(kwargs['labels'])
        outputs = self.model(input_ids)
        outputs["pooler_output"] = self.pooler(outputs["last_hidden_state"].permute(0, 2, 1)).flatten(start_dim=1)
        outputs["logits"] = self.head_sequence_classification(outputs["pooler_output"])
        
        try:
            outputs["loss"] = self.criterion(outputs["logits"], labels)
        else:
            outputs["loss"] = -1
        
        #
        """
        print(type(outputs))
        for k_, v_ in outputs.items():
            print(f"\n{k_}:")
            print(v_.shape)
        #
        print(f"\nlabels:")
        print(labels.shape)
        
        #
        print(outputs["loss"])
        """
        
        return outputs

def load_model_for_finetuning(
    run_name,
    config_name,
    num_labels,
    finetuning_task,
    cache_dir,
    revision,
    use_auth_token,
):
    #
    with open('map_.json', 'r') as f:
        map_ = json.load(f)
    
    #
    features = map_[run_name]
    
    #
    config = LanModelConfig(**features)
    #config.config_name = cofig_name
    config.num_labels = num_labels
    config.finetuning_task = finetuning_task
    config.cache_dir = cache_dir
    config.revision = revision
    config.use_auth_token = use_auth_token
    
    
    # define encoder 
    encoder = LanModel(config)
    
    #
    """
    for name, par_ in encoder.named_parameters():
        print(name)
        print(par_)
        break
    """
    
    # load weights
    path_read = f'./output_dir/{run_name}/best_model'
    encoder = encoder.from_pretrained(path_read)
    config.model = encoder
    
    #
    """
    for name, par_ in encoder.named_parameters():
        print(name)
        print(par_)
        break
    """
    
    #
    model_sequence_classification = LanModelSequenceClassification(config)
    
    return config, model_sequence_classification
    