# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import time
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

from typing import Callable, List, Optional, Union
from .ppo_trainer import PPOTrainer
from . import AdaptiveKLController, BaseTrainer, FixedKLController, PPOConfig
from ..models import PreTrainedModelWrapper
from ..core import logprobs_from_logits, stack_dicts, WANDB_PADDING, convert_to_scalar

class PPOVisionTrainer(PPOTrainer):
    """
    The PPOVisionTrainer uses Proximal Policy Optimization to optimise vision models.
    Note, this trainer is heavily inspired by the original OpenAI learning to summarize work here:
    https://github.com/openai/summarize-from-feedback

    Attributes:
        **config** (`PPOConfig`) -- Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more
         details.
        **model** (`PreTrainedModelWrapper`) -- Model to be optimized, Hugging Face transformer model with a value head.
            Check the documentation of `PreTrainedModelWrapper` for more details.
        **ref_model** (`PreTrainedModelWrapper`, *optional*) -- Reference model to be used for KL penalty, Hugging Face
            transformer model with a casual language modelling head. Check the documentation of `PreTrainedModelWrapper`
            for more details. If no reference model is provided, the trainer will create a reference model with the same
             architecture as the model to be optimized with shared layers.
        **tokenizer** (`Union[PreTrainedTokenizer, PreTrainedTokenizerFast]`) -- Tokenizer to be used for encoding the
            data. Check the documentation of `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*) -- PyTorch dataset or Hugging
            Face dataset. This is used to create a PyTorch dataloader. If no dataset is provided, the dataloader must be
             created outside the trainer users needs to design their own dataloader and make sure the batch
            size that is used is the same as the one specified in the configuration object.
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- Optimizer to be used for training. If no optimizer is
            provided, the trainer will create an Adam optimizer with the learning rate specified in the configuration
            object.
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- Data collator to be used for training and
            passed along the dataloader
        **num_shared_layers** (int, *optional*) -- Number of layers to be shared between the model and the reference
            model, if no reference model is passed. If no number is provided, all the layers will be shared.
        **lr_scheduler** (`torch.optim.lr_scheduler`, *optional*) -- Learning rate scheduler to be used for training.
        **processor** (Callable, *optional*) -- Callable to be used for processing the data. 
    """
    def __init__(self, 
        config: PPOConfig = None, 
        model: PreTrainedModelWrapper = None, 
        ref_model: PreTrainedModelWrapper = None, 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, 
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None, 
        optimizer: Optional[torch.optim.Optimizer] = None, 
        data_collator=None, 
        num_shared_layers: Optional[int] = None, 
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        processor = None,
    ):
        super().__init__(config, model, ref_model, tokenizer, dataset, optimizer, data_collator, num_shared_layers, lr_scheduler)
    
        self.processor = processor
        self.is_vision_ppo = True
    
    def step(
        self,
        pixel_values,
        labels,
        scores,
    ):
        timing = dict()
        t0 = time.time()
        # Step 1: get the reference logits and active logits
        with torch.no_grad():
            logits, _, values = self.model(pixel_values)
            ref_logits, _, _ = self.ref_model(pixel_values)

        all_logprobs = logprobs_from_logits(logits, labels)
        ref_logprobs = logprobs_from_logits(ref_logits, labels)

        masks = torch.ones_like(all_logprobs)
    

        # Step 2: compute the advantages
        rewards, non_score_reward = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)

        # Step 3: PPO
        all_stats = []
        for _ in range(self.config.ppo_epochs):
            logits, _, vpreds = self.model(pixel_values)
            logprobs = logprobs_from_logits(logits, labels)

            train_stats = self.train_minibatch(
                all_logprobs,
                values,
                rewards,
                logprobs,
                logits,
                vpreds,
                masks.unsqueeze(-1),
            )

            all_stats.append(train_stats)
        
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
        )

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(stats["objective/kl"].cpu(), self.config.batch_size * self.accelerator.num_processes)
        
        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats
