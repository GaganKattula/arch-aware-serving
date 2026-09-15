import importlib.metadata
import sys
import time
from importlib.metadata import PackageNotFoundError
from pathlib import Path

from huggingface_hub import snapshot_download

import torch
import transformers

print("py", sys.version.split()[0], "| torch", torch.__version__, flush=True)

from transformers import AutoTokenizer, Qwen3_5ForCausalLM

M = "Qwen/Qwen3.5-0.8B"
m = Qwen3_5ForCausalLM.from_pretrained(M, dtype=torch.bfloat16, attn_implementation="eager")

t0 = time.time(); m = m.to("mps"); m.eval()

# load tokenizer
tok = AutoTokenizer.from_pretrained(M)
ids = tok("The capital of France is", return_tensors="pt").input_ids.to("mps")


with torch.no_grad():
    out = m(ids, output_hidden_states=True)

ids2 = tok("Much of the framework that Fischbach teaches is essentially about making the plan before you set out to solve anything. But it’s rarely so straightforward as following your plan from A to Z. “When you start out thinking that your original plan is going to unfold exactly, that is an illusion. There’s no real project I’ve seen that doesn’t go through some serious twists and turns,” said Fischbach.", return_tensors="pt").input_ids.to("mps")

print( f"Confirming long prompt token count: {ids2.shape[1]} ")

with torch.no_grad():
    out2 = m(ids2, output_hidden_states=True)




def kernel_version_finder(names: list[str]):
    
    kernels= {}
    for name in names:
        try:
            ver = importlib.metadata.version(name)
            kernels[name]= ver

        except PackageNotFoundError:
            kernels[name]= None

    return kernels

        
kernels_installed = kernel_version_finder(["fla-core", "kernels", "triton"])

        
logits_1 = out.logits
logits_2 = out2.logits

hidden_states_1 = out.hidden_states
hidden_states_2 = out2.hidden_states


metadata = {"model_id" : "Qwen/Qwen3.5-0.8B",
            # NOT m.config._commit_hash — that is cleared during model load and
            # comes back None. The cached snapshot dir is named for the commit.
            "revision":  Path(snapshot_download(M, allow_patterns=["config.json"])).name,
            "dtype": "bfloat16",
            "attn_implementation": "eager",
            "device": "mps",
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "kernels installed": kernels_installed,
            "prompts": ["The capital of France is", "Much of the framework that Fischbach teaches is essentially about making the plan before you set out to solve anything. But it’s rarely so straightforward as following your plan from A to Z. When you start out thinking that your original plan is going to unfold exactly, that is an illusion. There’s no real project I’ve seen that doesn’t go through some serious twists and turns, said Fischbach."],
            "token_ids": [ids.tolist(), ids2.tolist()] }


# --- save the fixture -------------------------------------------------------
# .cpu() everything: an MPS-resident tensor makes the fixture device-tied, and
# these same files get compared on CUDA later.
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

reference = {
    "short": {
        "logits": logits_1.cpu(),
        "hidden_states": [h.cpu() for h in hidden_states_1],
    },
    "long": {
        "logits": logits_2.cpu(),
        "hidden_states": [h.cpu() for h in hidden_states_2],
    },
    "metadata": metadata,
}

out_path = FIXTURES / "qwen3_5_0_8b_reference.pt"
torch.save(reference, out_path)
print(f"saved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
print(f"  hidden_states per prompt: {len(reference['short']['hidden_states'])} tensors")
print(f"  logits shapes: short {tuple(logits_1.shape)}  long {tuple(logits_2.shape)}")
