from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# instantiate pipeline
pipeline = FlaxWhisperPipline("../models/whisper-large-v3/", dtype=jnp.bfloat16)

# JIT compile the forward call - slow, but we only do once
text = pipeline("../librispeech_dummy.wav")

# used cached function thereafter - super fast!!
print(text)
text = pipeline("../librispeech_dummy.wav")