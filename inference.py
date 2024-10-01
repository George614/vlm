from PIL import Image
import torch
import fire

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


def move_inputs_to_device(inputs: dict, device: str):
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_path: str, device: str
):
    image = Image.open(image_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs
    
    
def _sample_top_p(probs: torch.Tensor, top_p: float = 0.9):
    # [batch_size, vocab_size]
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # [batch_size, vocab_size]
    probs_cum = torch.cumsum(probs_sort, dim=-1)
    # [batch_size, vocab_size]
    # subtracting probs_sort shifts the cumsum by 1 position to the right before masking
    mask = probs_cum - probs_sort > top_p
    # zero out all the probabilities of tokens that are not selected by top_p
    probs_sort[mask] = 0.0
    # re-normalize the probabilities
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # sample the token index from the modified distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, dim=-1, index=next_token)
    return next_token
    

def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    model_inputs = get_model_inputs(processor, prompt, image_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    kv_cache = KVCache()
    # generate tokens until the stop token is generated
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    
    for _ in range(max_tokens_to_generate):
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]  # take the last logit along the sequence axis
        
        if do_sample:
            # apply tempeature and top-p sampling
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # remove the batch dimension
        generated_tokens.append(next_token)
        if next_token.item() == stop_token:
            break
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(  # attened to all previous tokens
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )
    
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    # decode the generated tokens
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print(prompt + decoded)


def main(
    model_path: str = None,
    prompt: str = 'A beautiful sunset over the mountains',
    image_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    device = "cpu"
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    print("Device in use:", device)
    print("Loading model from ", model_path)
    
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference...")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )

if __name__ == "__main__":
    fire.Fire(main)