# GPT-4o System Card: From Pipeline to End-to-End Multimodal AI

**Presenter:** Laura Li
**Course:** DS 5690-01 Gen AI Models in Theory & Practice  
**Date:** 11.13.2025

**Paper:** "GPT-4o System Card" by OpenAI (August 8, 2024)  
**Citation:** OpenAI. (2024). GPT-4o System Card. arXiv:2410.21276v1 [cs.CL]. https://arxiv.org/abs/2410.21276

---

## Overview (5 minutes)

### The Problem GPT-4o Addresses

Prior to GPT-4o, multimodal AI systems relied on **pipeline architectures** where separate models handled different modalities:

1. **Audio-to-Text Model** → transcribes speech
2. **Text-based LLM (GPT-3.5/GPT-4)** → processes and generates text
3. **Text-to-Audio Model** → synthesizes speech

**Key Limitations of Pipeline Approach:**
- **Information loss** at each conversion step (tone, emotion, multiple speakers, background noise)
- **High latency**: 2.8s (GPT-3.5) to 5.4s (GPT-4) response times
- **Loss of paralinguistic features**: Cannot directly observe or generate laughter, singing, emotional expression
- **Computational overhead**: Running three separate models increases costs and complexity

### GPT-4o's Revolutionary Approach

GPT-4o ("o" for "omni") is trained **end-to-end across text, vision, and audio** using a **single neural network** that:
- Accepts any combination of text, audio, image, and video as input
- Generates any combination of text, audio, and image as output
- Responds in 232-320ms (similar to human reaction time)
- Preserves all information across modalities

**This is fundamentally different from previous multimodal models** which used early fusion, late fusion, or cross-attention between separate encoders.

### How the Problem Was Addressed

1. **Unified Architecture**: Single transformer processes all modalities simultaneously
2. **Joint Training**: Pre-trained on multimodal data (text, images, audio, video) concurrently until October 2023
3. **Safety-First Design**: Extensive red teaming with 100+ experts across 29 countries
4. **Post-Training Alignment**: RLHF and safety mitigations integrated throughout
5. **Preparedness Framework**: Evaluated across cybersecurity, CBRN, persuasion, and model autonomy

---

## Architecture Overview: Transformer to Omni Model

### Question 1 for the Class (2 minutes)

**Consider the basic transformer architecture we studied in "Formal Algorithms for Transformers":**

In the standard decoder-only transformer (like GPT-2/GPT-3), we have:
- Token embeddings: `E_T ∈ ℝ^(d_e × V)` where V is vocabulary size
- Positional embeddings: `E_P ∈ ℝ^(d_e × L_max)`
- Input: `X = E_T[:, t[i]] + E_P[:, i]` for each token

**Question:** How would you need to modify the embedding layer to handle audio waveforms and images in addition to text tokens? What challenges arise when trying to process all three modalities in a single transformer?

**Give the class 1-2 minutes to think/discuss**

<details>
<summary><b>Discussion Points</b></summary>

- Audio: Need to convert continuous waveform → discrete tokens (e.g., via audio encoder like Whisper)
- Images: Need to convert 2D pixels → token sequences (e.g., via Vision Transformer patches)
- Alignment problem: How to synchronize temporal information across modalities?
- Vocabulary: Unified token space vs. separate token spaces per modality
- Context window: Audio/video consume much more tokens than text

</details>

---

### Question 2 for the Class (1 minute)

**In standard transformers, we use:**
```
Mask[i,j] = 1 if i ≤ j (unidirectional attention)
```

**Question:** In a conversation where GPT-4o receives audio input and generates audio output in real-time, how should attention masking work when audio tokens are being streamed in and out simultaneously?

<details>
<summary><b>Discussion Points</b></summary>

- Need causal masking for output generation
- But full attention over received input tokens
- Handling interruptions: What happens to attention if user interrupts?
- Streaming vs. batch processing trade-offs

</details>

---

## Formal Architecture Description

### From Standard Transformer to GPT-4o

Based on "Formal Algorithms for Transformers" (Phuong & Hutter, 2022), let's formally describe how GPT-4o extends the decoder-only transformer architecture:

#### Standard Decoder-Only Transformer (GPT-2/GPT-3)

```
Algorithm: DTransformer(s|θ)
Input: s ∈ V*, sequence of token IDs
Output: P ∈ (0,1)^(V×|s|), probability distribution over next tokens

Parameters:
  - E_T ∈ ℝ^(d_e × V): token embedding matrix
  - E_P ∈ ℝ^(d_e × L_max): positional embedding matrix
  - For layer l ∈ [L]:
    - W_l: multi-head attention parameters
    - γ¹_l, β¹_l, γ²_l, β²_l ∈ ℝ^d_e: layer norm parameters
    - W_mlp1, W_mlp2: MLP parameters
  - E_U ∈ ℝ^(V × d_e): unembedding matrix

1. n ← length(s)
2. For i ∈ [n]: x_i ← E_T[:, s[i]] + E_P[:, i]
3. X ← [x_1, x_2, ..., x_n]
4. For l = 1, 2, ..., L:
5.   X̃ ← LayerNorm(X | γ¹_l, β¹_l)
6.   X ← X + MHAttention(X̃ | W_l, Mask[i,j] = [[i ≤ j]])
7.   X̃ ← LayerNorm(X | γ²_l, β²_l)
8.   X ← X + MLP(X̃)
9. X ← LayerNorm(X | γ, β)
10. Return P = softmax(E_U · X)
```

#### GPT-4o Omni Model Extension

```
Algorithm: OmniTransformer(I_text, I_audio, I_image, I_video | θ)
Input: 
  - I_text ∈ V_text*: text token sequence
  - I_audio ∈ ℝ^(T_audio × d_audio): audio waveform
  - I_image ∈ ℝ^(H × W × 3): image
  - I_video ∈ ℝ^(F × H × W × 3): video frames
  
Output: 
  - O_text ∈ (0,1)^(V_text × n_text): text probability distribution
  - O_audio ∈ ℝ^(T_out × d_audio): generated audio waveform
  - O_image ∈ ℝ^(H_out × W_out × 3): generated image

Parameters:
  - E_text ∈ ℝ^(d_e × V_text): text embedding
  - φ_audio: audio encoder (e.g., Whisper-style)
  - φ_vision: vision encoder (e.g., CLIP ViT-style)  
  - For layer l ∈ [L]:
    - W_l: unified multi-head attention (works across all modalities)
    - γ¹_l, β¹_l, γ²_l, β²_l ∈ ℝ^d_e: layer norm parameters
    - W_mlp1, W_mlp2: MLP parameters
  - ψ_text: text output head
  - ψ_audio: audio decoder
  - ψ_image: image decoder

# Stage 1: Modality-Specific Encoding
1. X_text ← [E_text[:, I_text[i]] + E_P[:, i] for i ∈ [n_text]]
2. X_audio ← φ_audio(I_audio)  # Maps to token-like representations
3. X_image ← φ_vision(I_image)  # Converts to patch tokens
4. X_video ← [φ_vision(I_video[f]) for f ∈ [F]]  # Process each frame

# Stage 2: Concatenate Multi-Modal Input
5. X ← Concat([X_text, X_audio, X_image, X_video])
6. n_total ← length(X)

# Stage 3: Unified Transformer Processing
7. For l = 1, 2, ..., L:
8.   X̃ ← LayerNorm(X | γ¹_l, β¹_l)
9.   # Attention now operates across ALL modalities
10.  X ← X + MHAttention(X̃ | W_l, Mask[i,j] = [[i ≤ j]])
11.  X̃ ← LayerNorm(X | γ²_l, β²_l)
12.  X ← X + MLP(X̃)

# Stage 4: Modality-Specific Decoding
13. X_final ← LayerNorm(X | γ, β)
14. O_text ← ψ_text(X_final)  # Project to text vocabulary
15. O_audio ← ψ_audio(X_final)  # Decode to audio waveform
16. O_image ← ψ_image(X_final)  # Decode to image pixels

17. Return (O_text, O_audio, O_image)
```

### Key Architectural Differences from Standard Transformers

| Component | Standard Transformer | GPT-4o Omni |
|-----------|---------------------|-------------|
| **Input Embedding** | Single token embedding matrix | Multiple encoders (text, audio, vision) → unified representation |
| **Attention** | Over token sequence | Over concatenated multi-modal token sequence |
| **Training** | Text-only pre-training | Simultaneous pre-training on text, audio, image, video |
| **Output** | Single modality (text tokens) | Multiple modalities (text, audio, image) generated in parallel |
| **Information Flow** | Token → Token | Multi-modal tokens → Multi-modal outputs (no information loss) |
| **Latency** | N/A for text, 2.8-5.4s for voice mode | 232-320ms end-to-end for audio |

### What Makes This Different from Prior Multimodal Models?

**GPT-4 with Vision (March 2023):**
- Image encoder → text LLM → text output only
- Still a pipeline: vision features fed as additional context to text model

**Previous Voice Mode:**
- ASR model → GPT-4 → TTS model
- Three separate models, information loss at boundaries

**GPT-4o:**
- **Single neural network** processes everything end-to-end
- No modality conversion until final output
- Can attend to audio patterns, visual features, and text simultaneously

---

## Critical Analysis

### What Was Overlooked or Could Be Developed Further?

#### 1. **Lack of Architectural Details**

**The Paper States:**
> "GPT-4o is trained end-to-end across text, vision, and audio, meaning that all inputs and outputs are processed by the same neural network."

**What's Missing:**
- **Exact tokenization scheme** for audio (discrete tokens? continuous embeddings?)
- **Vision encoder architecture** (CLIP-style? How many patches?)
- **Audio decoder design** (autoregressive? diffusion-based? streaming?)
- **How modalities are synchronized** temporally
- **Model size**: Number of parameters not disclosed (likely ~1.76T based on leaks, but unconfirmed)

This lack of transparency makes it **impossible to reproduce** or verify claims. Compare this to "Formal Algorithms for Transformers" which provides complete pseudocode.

#### 2. **Training Data Composition Not Disclosed**

The paper mentions training on:
- Web data
- Code and math
- Multimodal data (images, audio, video)

**Missing:**
- Exact proportions of each modality
- Audio data sources and preprocessing
- Video data quantity (critical for learning temporal dynamics)
- How they ensured data quality across modalities

**Why This Matters:** The ratio of text:audio:video data fundamentally affects what the model learns. Too little audio data and it won't generalize well to diverse accents/languages.

#### 3. **Evaluation Methodology Has Significant Limitations**

The paper acknowledges:
> "We used Voice Engine to convert text inputs to audio, feed it to GPT-4o, and score the outputs by the model."

**Problems:**
1. **Circular dependency**: Evaluating speech using TTS-generated speech may not represent real human speech
2. **Missing edge cases**: Background noise, multiple speakers, cross-talk, non-native accents
3. **Text-centric bias**: Many benchmarks are designed for text and may not capture audio-specific capabilities

**Better approach**: Evaluate on naturally occurring multi-modal conversations (e.g., podcast transcription + comprehension).

#### 4. **Anthropomorphization Risks Identified But Not Fully Mitigated**

The paper notes users saying:
> "This is our last day together."

**Critical Issue:** This suggests users are forming **parasocial relationships** with the AI. While acknowledged, the mitigations are unclear:
- No discussion of warning users about anthropomorphization
- No "friction" introduced to remind users they're talking to an AI
- Preset voices may make this worse by sounding very human-like

**Recommendation:** Add periodic reminders ("I'm Claude, an AI assistant") or subtle indicators that break the illusion of human conversation.

#### 5. **Voice Cloning Threat Model Incomplete**

**Mitigation:**
- Preset voices only
- Output classifier to detect voice deviation (100% recall claimed)

**What's Missing:**
- What about **adversarial attacks** that fool the classifier?
- How robust is it to **synthetic voices** designed to mimic but not exactly match?
- **Zero-shot voice cloning** from text instructions ("sound like a gruff detective")

The paper doesn't test adversarial robustness of the voice classifier.

### Have Others Disputed the Findings?

#### **METR's Independent Assessment** (Section 4.1)

METR found:
> "They did not find a significant increase in these capabilities for GPT-4o as compared to GPT-4."

For autonomous capabilities, **GPT-4o showed no major advancement** over GPT-4. This suggests that multimodality alone doesn't improve agentic reasoning—contradicting some hype around "omni" models.

#### **Apollo Research on Scheming** (Section 4.2)

> "GPT-4o showed moderate self-awareness...but lacked strong capabilities in reasoning about itself or others in applied agent settings."

This suggests GPT-4o is **not yet capable of deceptive scheming**, which is reassuring but also indicates current evaluation methods may not detect more subtle forms of misalignment.

#### **The Mini-Omni2 Open Source Replication**

Researchers at [Mini-Omni2](https://arxiv.org/abs/2410.11190) attempted to replicate GPT-4o's architecture and found:

**Challenges:**
1. "Substantial data requirements—training for GPT-4o necessitates the integration of data across visual, audio, and textual modalities, with quantities increasing exponentially"
2. "Direct inference output capabilities in multi-modal contexts" remain very difficult
3. Had to use **pre-trained encoders** (CLIP + Whisper) rather than truly end-to-end training

This suggests GPT-4o's achievement required **far more computational resources** than OpenAI disclosed, making it difficult for open research to replicate.

### Errors or Inconsistencies?

1. **Persuasion Risk Rated "Medium"** (borderline), but audio persuasion rated "Low" 
   - This seems inconsistent: if text is borderline medium, shouldn't audio (which is more persuasive) also be medium?

2. **100% Recall Claimed** for voice classifier, but "rare instances where the model would unintentionally generate an output emulating the user's voice"
   - If it happens, even rarely, recall isn't 100%
   - Likely they mean 100% on *test set*, but not in deployment

3. **Capability evaluations show improvement over GPT-4 Turbo** (Table 7 for medical benchmarks), but METR found "no significant increase" in autonomous capabilities
   - This suggests GPT-4o is better at **passive reasoning** but not **active agency**

---

## Impact Analysis

### How Did/Does GPT-4o Change the AI Landscape?

#### 1. **Paradigm Shift: From Pipeline to End-to-End**

**Before GPT-4o:**
- Multimodal AI = ensemble of specialized models
- DALL-E for images, Whisper for speech, GPT-4 for text
- Each model trained separately, combined via APIs

**After GPT-4o:**
- Single model that "natively understands" multiple modalities
- Sets precedent: future models will be omni by default
- Other labs rushing to match: Google Gemini 1.5, Anthropic Claude 3 Opus (vision), Meta's upcoming omni models

#### 2. **Democratization of Multimodal AI**

**Impact:**
- GPT-4o is **50% cheaper** and **2x faster** than GPT-4 Turbo
- Available to **free tier users** (unprecedented for such a capable model)
- API rate limits 5x higher than GPT-4 Turbo

**Result:** Small startups can now build multimodal applications that were previously cost-prohibitive.

#### 3. **Real-Time Voice Interaction Becomes Viable**

**232-320ms latency** enables:
- **Live translation** with human-like responsiveness
- **Conversational tutoring** (math, language learning)
- **Accessibility tools** for blind/low-vision users (real-time scene description)

**Example:** Duolingo's [Roleplay feature](https://blog.duolingo.com/duolingo-max/) uses GPT-4o for conversational language practice.

#### 4. **New Safety Challenges**

GPT-4o introduced risks that didn't exist before:

| Risk | Why It Matters | Mitigation |
|------|----------------|------------|
| **Voice cloning** | Could enable fraud, impersonation | Preset voices only, real-time classifier |
| **Speaker identification** | Privacy risk (identifying people by voice) | Trained to refuse identification requests |
| **Emotional manipulation** | Human-like voice may be more persuasive | Studied persuasiveness (found Low risk) |
| **Anthropomorphization** | Users forming attachments ("This is our last day together") | Under study, no clear mitigation yet |
| **Unauthorized voice generation** | Model might mimic user's voice unintentionally | Output classifier (100% recall claimed) |

These risks are unique to **speech-to-speech AI** and weren't considerations for text-only models.

### Intersection with Other Work

#### **Past: Foundations**

- **Transformer architecture** (Vaswani et al., 2017) - the foundation
- **GPT-3** (Brown et al., 2020) - showed scale enables few-shot learning
- **CLIP** (Radford et al., 2021) - vision-language pre-training
- **Whisper** (Radford et al., 2022) - robust speech recognition

GPT-4o **combines insights** from all of these into a unified model.

#### **Present: Competition**

| Model | Released | Key Feature | Limitation vs GPT-4o |
|-------|----------|-------------|---------------------|
| **Google Gemini 1.5** | Feb 2024 | 10M token context window | Not truly end-to-end (separate encoders) |
| **Anthropic Claude 3 Opus** | Mar 2024 | Vision + strong reasoning | No native audio |
| **Meta Llama 3.2** | Sep 2024 | Open source, vision | No audio, smaller scale |
| **Microsoft Phi-4** | Dec 2024 | Efficient small model | Text-only |

GPT-4o is the **only major model with native audio input/output**.

#### **Future: What's Next**

**Short-term (6-12 months):**
1. **Gemini 2.0** likely to add native audio
2. **Open-source omni models** (e.g., [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio))
3. **Mobile deployment** of omni models (Apple Intelligence rumors)

**Long-term (1-3 years):**
1. **Continuous multimodal learning**: Models that can be updated with new modalities without full retraining
2. **Grounded omni models**: Integration with robotics (visual + audio + action)
3. **Efficiency breakthroughs**: Current omni models are very expensive to train

**Scientific acceleration** (Section 5.3):
- Omni models could enable "AI research assistants" that read papers, understand figures, and discuss ideas via voice
- Red teamers found GPT-4o could understand "research-level quantum physics"
- Potential to accelerate science by helping with "mundane tasks" (literature review, figure interpretation)

### Importance and Broader Implications

**Why GPT-4o Matters:**

1. **Technical Achievement**: Proves that end-to-end multimodal learning is feasible at scale
2. **Accessibility**: Makes advanced AI more accessible (free tier, lower cost)
3. **New Use Cases**: Enables applications impossible with pipeline architectures
4. **Safety Precedent**: Sets standard for how to evaluate and deploy multimodal AI safely

**But Also:**

1. **Centralization**: Only OpenAI, Google, Anthropic can afford to train these models
2. **Black Box**: Lack of architectural details hampers research
3. **New Risks**: Anthropomorphization, voice cloning, emotional manipulation
4. **Evaluation Gap**: We don't yet know how to properly evaluate omni models

---

## Additional Resources

1. **Official Paper**: [GPT-4o System Card (arXiv)](https://arxiv.org/abs/2410.21276) - Full technical and safety evaluation details

2. **OpenAI Blog Post**: [Hello GPT-4o](https://openai.com/index/hello-gpt-4o/) - High-level introduction with demo videos

3. **Third-Party Evaluations**:
   - [METR's GPT-4o Assessment](https://metr.org/blog/2024-08-gpt-4o/) - Independent evaluation of autonomous capabilities
   - [Apollo Research on Scheming](https://www.apolloresearch.ai/research/evaluating-gpt-4o-for-scheming) - Tests for deceptive behavior

4. **Open Source Replications**:
   - [Mini-Omni2 Paper](https://arxiv.org/abs/2410.11190) - Attempt to recreate GPT-4o architecture
   - [Mini-Omni2 GitHub](https://github.com/gpt-omni/mini-omni2) - Code and model weights

5. **Related Technical Deep Dives**:
   - [Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238) - Foundational transformer architecture (the paper we studied in class)
   - [Multimodal Learning Overview](https://dataroots.io/blog/gpt4-multimodality) - Explains fusion techniques (early, late, cross-attention)

6. **Safety and Ethics**:
   - [OpenAI Red Teaming Network](https://openai.com/index/red-teaming-network/) - How external experts evaluate models
   - [Anthropomorphization Risks](https://www.technologyreview.com/2024/08/09/1094715/openai-gpt4o-emotional-voice/) - MIT Tech Review on emotional reliance

7. **Comparative Analysis**:
   - [GPT-4o vs Gemini 1.5 Pro](https://artificialanalysis.ai/models/gpt-4o/compare) - Benchmark comparison
   - [Roboflow's GPT-4o Vision Guide](https://blog.roboflow.com/gpt-4o-vision-use-cases/) - Computer vision use cases

```

### Key Observations from Code:

1. **API Design**: GPT-4o's API treats images/audio as first-class inputs alongside text
2. **Streaming**: Real-time response generation is built-in
3. **Limitation**: Full audio generation not yet in public API (still in limited alpha)
4. **Backwards Compatible**: Can fall back to pipeline approach (Whisper + GPT-4o + TTS) if needed

---

## Summary

**GPT-4o represents a fundamental architectural shift** in how we build multimodal AI systems:

1. **From Pipeline to End-to-End**: Single neural network processes all modalities, eliminating information loss
2. **From Specialized to Unified**: One model handles text, audio, vision, and video simultaneously
3. **From Slow to Real-Time**: 232-320ms latency enables human-like conversational interaction
4. **From Expensive to Accessible**: 50% cheaper, 2x faster, available to free users

**Key Technical Innovations:**
- End-to-end training across modalities (text, audio, vision, video)
- Unified attention mechanism operating on concatenated multi-modal tokens
- Real-time audio generation (not just transcription)
- Extensive safety mitigations for voice-specific risks

**Critical Gaps:**
- Architecture details not disclosed (model size, tokenization, decoder design)
- Training data composition unknown
- Evaluation methods have limitations (TTS-based testing)
- Anthropomorphization risks acknowledged but not fully addressed

**Impact:**
- **Near-term**: Enables new applications (live translation, conversational tutoring, accessibility)
- **Long-term**: Sets paradigm for next generation of foundation models
- **Research**: Opens questions about evaluation, safety, and optimal multimodal architectures

**Future Directions:**
- More efficient training methods for omni models
- Better evaluation frameworks for multimodal capabilities
- Open-source replications to enable broader research
- Integration with robotics and embodied AI

---

**Questions for Discussion:**
1. Can we trust benchmarks designed for text when evaluating multimodal models?
2. How should we balance the benefits of human-like AI interaction against anthropomorphization risks?
3. Will the "omni model" paradigm dominate, or will specialized models remain competitive?
4. What architectural innovations could make omni models more sample-efficient to train?

---

**Acknowledgments:** Thanks to OpenAI for releasing the system card, METR and Apollo Research for independent evaluations, and the open-source community working on replication efforts.
