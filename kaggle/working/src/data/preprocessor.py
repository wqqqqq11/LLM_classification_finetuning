"""Text preprocessing with smart truncation for LLM classification.

Implements the truncation strategy from plan_v1.md:
- Max length: 3072 tokens
- Template reserved: ~150 tokens
- Prompt max: 512 tokens (head 40% + tail 60%)
- Response max: 1200 tokens each (head 65% + tail 35%)
- Dynamic compensation between prompt and responses
"""

from typing import Dict, Tuple
from transformers import PreTrainedTokenizer


class TextPreprocessor:
    """Handles text formatting and token budget management."""
    
    TEMPLATE = """You are judging which assistant response a human user would prefer.
The order of responses is arbitrary. Judge only the content.

[User Prompt]
{prompt}

[Response A]
{response_a}

[Response B]
{response_b}

Predict the human preference:
A = Response A is better
B = Response B is better
Tie = both responses are similarly preferred

Preference:"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 3072,
        template_tokens: int = 150,
        prompt_max_tokens: int = 512,
        response_max_tokens: int = 1200,
        prompt_head_ratio: float = 0.4,
        prompt_tail_ratio: float = 0.6,
        response_head_ratio: float = 0.65,
        response_tail_ratio: float = 0.35,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_tokens = template_tokens
        self.prompt_max_tokens = prompt_max_tokens
        self.response_max_tokens = response_max_tokens
        self.prompt_head_ratio = prompt_head_ratio
        self.prompt_tail_ratio = prompt_tail_ratio
        self.response_head_ratio = response_head_ratio
        self.response_tail_ratio = response_tail_ratio
        
        # Calculate available tokens for content
        self.available_tokens = max_length - template_tokens
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def truncate_text(
        self,
        text: str,
        max_tokens: int,
        head_ratio: float,
        tail_ratio: float,
    ) -> str:
        """Truncate text keeping head and tail portions.
        
        Args:
            text: Input text
            max_tokens: Maximum tokens allowed
            head_ratio: Ratio of tokens to keep from head
            tail_ratio: Ratio of tokens to keep from tail
            
        Returns:
            Truncated text
        """
        if not text or not isinstance(text, str):
            return ""
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Calculate head and tail token counts
        head_tokens = int(max_tokens * head_ratio)
        tail_tokens = int(max_tokens * tail_ratio)
        
        # Adjust to ensure we don't exceed max
        total = head_tokens + tail_tokens
        if total > max_tokens:
            tail_tokens = max_tokens - head_tokens
        
        # Extract head and tail
        head_ids = tokens[:head_tokens]
        tail_ids = tokens[-tail_tokens:] if tail_tokens > 0 else []
        
        # Decode and join with separator
        head_text = self.tokenizer.decode(head_ids, skip_special_tokens=True)
        tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=True) if tail_ids else ""
        
        if tail_text:
            return head_text.strip() + "\n... [truncated] ...\n" + tail_text.strip()
        return head_text.strip()
    
    def allocate_token_budget(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> Tuple[int, int, int]:
        """Allocate token budget dynamically between prompt and responses.
        
        Core principle: response_a and response_b must have equal initial budget
        to avoid length bias.
        
        Returns:
            (prompt_budget, response_a_budget, response_b_budget)
        """
        # Count actual tokens
        prompt_tokens = self.count_tokens(prompt)
        response_a_tokens = self.count_tokens(response_a)
        response_b_tokens = self.count_tokens(response_b)
        
        # Start with maximum allowed
        prompt_budget = self.prompt_max_tokens
        response_budget = self.response_max_tokens
        
        # Calculate remaining after prompt
        prompt_actual = min(prompt_tokens, prompt_budget)
        remaining = self.available_tokens - prompt_actual
        
        # Split remaining equally between responses
        per_response_budget = remaining // 2
        
        # If per_response_budget exceeds max, cap it and give back to prompt
        if per_response_budget > self.response_max_tokens:
            extra = (per_response_budget - self.response_max_tokens) * 2
            per_response_budget = self.response_max_tokens
            prompt_budget = min(self.prompt_max_tokens + extra, self.available_tokens - 2 * per_response_budget)
            response_a_budget = per_response_budget
            response_b_budget = per_response_budget
        else:
            # If responses don't use their budget, give back to prompt
            response_a_actual = min(response_a_tokens, per_response_budget)
            response_b_actual = min(response_b_tokens, per_response_budget)
            
            # Recalculate with actual usage
            response_a_budget = per_response_budget
            response_b_budget = per_response_budget
            
            unused_a = per_response_budget - response_a_actual
            unused_b = per_response_budget - response_b_tokens if response_b_tokens < per_response_budget else 0
            
            # Redistribute unused tokens
            total_unused = unused_a + unused_b
            if total_unused > 0:
                # Give to the longer response first, then to prompt
                if response_a_tokens > per_response_budget and response_b_tokens <= per_response_budget:
                    give_to_a = min(unused_b, response_a_tokens - per_response_budget)
                    response_a_budget += give_to_a
                    unused_b -= give_to_a
                elif response_b_tokens > per_response_budget and response_a_tokens <= per_response_budget:
                    give_to_b = min(unused_a, response_b_tokens - per_response_budget)
                    response_b_budget += give_to_b
                    unused_a -= give_to_b
                
                # Remaining goes to prompt
                prompt_budget = min(
                    self.prompt_max_tokens + (unused_a + unused_b) // 2,
                    self.available_tokens - response_a_budget - response_b_budget
                )
            else:
                prompt_budget = self.prompt_max_tokens
                response_a_budget = per_response_budget
                response_b_budget = per_response_budget
        
        return prompt_budget, response_a_budget, response_b_budget
    
    def preprocess(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> Dict[str, str]:
        """Preprocess and format the input text.
        
        Args:
            prompt: User prompt
            response_a: First model response
            response_b: Second model response
            
        Returns:
            Dictionary with formatted text and components
        """
        # Handle None values
        prompt = prompt if isinstance(prompt, str) else ""
        response_a = response_a if isinstance(response_a, str) else ""
        response_b = response_b if isinstance(response_b, str) else ""
        
        # Allocate token budgets
        prompt_budget, resp_a_budget, resp_b_budget = self.allocate_token_budget(
            prompt, response_a, response_b
        )
        
        # Truncate each component
        truncated_prompt = self.truncate_text(
            prompt, prompt_budget,
            self.prompt_head_ratio, self.prompt_tail_ratio
        )
        
        truncated_response_a = self.truncate_text(
            response_a, resp_a_budget,
            self.response_head_ratio, self.response_tail_ratio
        )
        
        truncated_response_b = self.truncate_text(
            response_b, resp_b_budget,
            self.response_head_ratio, self.response_tail_ratio
        )
        
        # Format final text
        formatted_text = self.TEMPLATE.format(
            prompt=truncated_prompt,
            response_a=truncated_response_a,
            response_b=truncated_response_b,
        )
        
        return {
            "formatted_text": formatted_text,
            "prompt": truncated_prompt,
            "response_a": truncated_response_a,
            "response_b": truncated_response_b,
            "token_count": self.count_tokens(formatted_text),
        }
