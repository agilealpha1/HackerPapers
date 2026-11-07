"""HackerPapers topic taxonomy v0: 46 topics in 8 areas, matched by keyword rules.

Each topic matches a regex against the title (weight 3) and abstract (weight 1); listed arXiv
categories add 2. A paper gets its 1-3 best topics. Step 4 replaces this with an LLM-drafted
taxonomy; the ids are meant to stay stable.
"""
AREAS = [
    ('llm', 'Language models', [
        ('large-language-models', 'Large language models', r'\b(large language models?|llms?|gpt-?\d|chatgpt|llama|foundation models?|language models?)\b', []),
        ('scaling-laws', 'Scaling laws & compute', r'\b(scaling laws?|compute[- ]optimal|chinchilla|emergent abilit\w*|scaling up)\b', []),
        ('prompting-reasoning', 'Prompting & reasoning', r'\b(chain[- ]of[- ]thought|prompting|in[- ]context learning|few[- ]shot learners|reasoning (models?|traces?)|test[- ]time compute|self[- ]consistency)\b', []),
        ('alignment-finetuning', 'Fine-tuning & RLHF', r'\b(rlhf|human feedback|instruction[- ]tun\w*|fine[- ]tun\w*|dpo|direct preference|reward models?|constitutional ai|lora|low[- ]rank adaptation|peft)\b', []),
        ('rag-retrieval', 'Retrieval & RAG', r'\b(retrieval[- ]augmented|rag|dense retrieval|vector search|embedding models?|rerank\w*)\b', ['cs.IR']),
        ('efficient-inference', 'Efficient inference & quantization', r'\b(quantiz\w*|distillation|pruning|speculative decoding|kv[- ]cache|inference (efficien\w*|cost|speed)|1[- ]bit|bitnet|sparsity)\b', []),
        ('long-context-attention', 'Attention & long context', r'\b(attention mechanisms?|self[- ]attention|flash ?attention|long[- ]context|context windows?|linear attention|positional (encodings?|embeddings?)|rope)\b', []),
        ('code-generation', 'Code generation', r'\b(code generation|program synthesis|coding (assistants?|agents?)|copilot|codex|software engineering agents?|swe[- ]bench)\b', []),
        ('evaluation-benchmarks', 'Evaluation & benchmarks', r'\b(benchmarks?|evaluation of|leaderboards?|contamination|arc[- ]agi)\b', []),
        ('hallucination', 'Hallucination & factuality', r'\b(hallucinat\w*|factuality|truthful\w*|sycophan\w*)\b', []),
        ('classic-nlp', 'NLP: translation, parsing, embeddings', r'\b(machine translation|word2vec|word (vectors|embeddings)|glove|parsing|named entity|sentiment analysis|seq2seq|sequence to sequence)\b', ['cs.CL']),
    ]),
    ('agents', 'Agents & reasoning', [
        ('agents-tool-use', 'LLM agents & tool use', r'\b(agents?|agentic|tool use|toolformer|function calling|web navigation|computer use)\b', ['cs.MA']),
        ('planning-search', 'Planning & search', r'\b(planning|monte carlo tree search|mcts|tree search|heuristic search)\b', []),
        ('math-theorem-proving', 'Math & theorem proving', r'\b(theorem prov\w*|formal (proofs?|verification|mathematics)|lean 4|olympiad|mathematical reasoning|autoformaliz\w*)\b', []),
    ]),
    ('generative', 'Generative models', [
        ('diffusion', 'Diffusion models', r'\b(diffusion models?|denoising diffusion|score[- ]based|stable diffusion|latent diffusion|flow matching)\b', []),
        ('gans', 'GANs', r'\b(generative adversarial|gans?|stylegan|cyclegan)\b', []),
        ('vae-flows', 'VAEs & normalizing flows', r'\b(variational auto-?encoders?|vaes?|normalizing flows?|auto-encoding variational)\b', []),
        ('image-video-generation', 'Image & video generation', r'\b(text[- ]to[- ](image|video)|image generation|video generation|image synthesis|dall-?e|imagen|sora)\b', []),
        ('speech-audio', 'Speech & audio', r'\b(speech|audio|text[- ]to[- ]speech|tts|asr|whisper|music generation|wavenet|voice)\b', ['cs.SD', 'eess.AS']),
    ]),
    ('vision', 'Vision & multimodal', [
        ('computer-vision', 'Computer vision', r'\b(image classification|object detection|segmentation|imagenet|convolutional|resnet|vision transformers?|vit|yolo)\b', ['cs.CV']),
        ('3d-nerf', '3D, NeRF & splatting', r'\b(nerf|neural radiance|gaussian splatting|3d reconstruction|point clouds?|novel view)\b', []),
        ('multimodal', 'Vision-language & multimodal', r'\b(vision[- ]language|multimodal|multi-modal|clip|image captioning|visual question)\b', []),
    ]),
    ('rl', 'Reinforcement learning & robotics', [
        ('reinforcement-learning', 'Reinforcement learning', r'\b(reinforcement learning|policy gradients?|q[- ]learning|actor[- ]critic|ppo|deep rl)\b', []),
        ('games', 'Game-playing AI', r'\b(alphago|alphazero|muzero|atari|chess|poker|starcraft|dota|game[- ]playing|minecraft|game of go)\b', []),
        ('robotics', 'Robotics & embodied AI', r'\b(robots?|robotic\w*|embodied|manipulation|locomotion|sim[- ]to[- ]real|autonomous driving|self[- ]driving)\b', ['cs.RO']),
    ]),
    ('foundations', 'Foundations', [
        ('dl-theory-optimization', 'Deep learning theory & optimization', r'\b(generalization|optimization|gradient descent|sgd|adam|loss landscapes?|double descent|grokking|neural tangent|convergence|overparameteri\w*|lottery ticket)\b', ['stat.ML']),
        ('architectures', 'Architectures: Transformers, RNNs, SSMs', r'\b(transformers?|recurrent neural|lstms?|rnns?|state space models?|mamba|architectures?|neural turing|memory networks?|capsules?)\b', []),
        ('training-techniques', 'Training techniques', r'\b(batch normalization|layer norm\w*|dropout|initialization|residual|learning rates?|regularization|data augmentation|mixed precision)\b', []),
        ('mixture-of-experts', 'Mixture of experts', r'\b(mixture[- ]of[- ]experts|moe|sparse experts|switch transformers?)\b', []),
        ('self-supervised', 'Representation & self-supervised learning', r'\b(self[- ]supervised|contrastive|representation learning|pre-?training|masked (auto-?encoders?|language model\w*)|bert)\b', []),
        ('graph-learning', 'Graph neural networks', r'\b(graph neural|gnns?|graph convolution\w*|message passing|knowledge graphs?)\b', []),
        ('probabilistic-ml', 'Probabilistic & Bayesian ML', r'\b(bayesian|gaussian process\w*|variational inference|probabilistic programming|uncertainty|causal\w*)\b', ['stat.ML']),
        ('classical-ml', 'Classical ML', r'\b(support vector|random forests?|gradient boost\w*|xgboost|decision trees?|k-means|clustering|kernel methods?|tabular)\b', []),
        ('neuro-evolution', 'Evolutionary & neuroevolution', r'\b(evolutionary|genetic algorithms?|neuroevolution|evolution strateg\w*|open[- ]ended\w*)\b', ['cs.NE']),
    ]),
    ('safety', 'Safety, interpretability & society', [
        ('interpretability', 'Interpretability', r'\b(interpretab\w*|explainab\w*|mechanistic|circuits?|probing|superposition|sparse auto-?encoders?|feature visuali\w*|saliency|attribution)\b', []),
        ('ai-safety', 'AI safety & alignment', r'\b(ai safety|alignment|misalign\w*|existential|deceptive|power[- ]seeking|scheming|reward hacking|jailbreak\w*|red[- ]teaming)\b', []),
        ('adversarial-security', 'Adversarial attacks & security', r'\b(adversarial (examples?|attacks?|robustness)|prompt injection|backdoors?|poisoning|watermark\w*)\b', ['cs.CR']),
        ('privacy-federated', 'Privacy & federated learning', r'\b(differential privacy|privacy|federated|membership inference|memoriz\w*|extracting training data)\b', []),
        ('fairness-ethics', 'Fairness, bias & ethics', r'\b(fairness|bias(es|ed)?|ethics|ethical|stochastic parrots|discriminat\w*|harms?)\b', ['cs.CY']),
        ('ai-society', 'AI, work & society', r'\b(productivity|labou?r|economic\w*|jobs|developers?|education|students|society|policy|regulation)\b', ['cs.CY', 'econ.GN']),
    ]),
    ('systems', 'Systems, data & applications', [
        ('ml-systems', 'ML systems & hardware', r'\b(gpus?|tpus?|distributed training|parallelism|kernels?|compilers?|hardware|throughput|serving|accelerators?|cuda|memory bandwidth)\b', ['cs.DC', 'cs.AR', 'cs.PF', 'cs.DB', 'cs.OS']),
        ('data-datasets', 'Data & datasets', r'\b(datasets?|data curation|synthetic data|web[- ]scale data|common crawl|data quality|tinystories|textbooks are all you need)\b', []),
        ('recsys-search', 'Recommendation & search', r'\b(recommend(er|ation) systems?|collaborative filtering|learning to rank|search engines?)\b', ['cs.IR']),
        ('time-series', 'Time series & forecasting', r'\b(time series|forecasting|weather)\b', []),
        ('science-medicine', 'AI for science & medicine', r'\b(proteins?|alphafold|molecul\w*|drugs?|medical|clinical|biolog\w*|chemistry|materials|physics[- ]informed|genom\w*)\b', ['q-bio.BM', 'q-bio.QM', 'physics.comp-ph']),
        ('neuro-cognition', 'Neuroscience & intelligence', r'\b(brains?|neuroscien\w*|cognitive|consciousness|measure of intelligence|general intelligence|agi|human-like)\b', ['q-bio.NC']),
    ]),
]

TOPICS = [dict(id=t[0], name=t[1], area=a[0], area_name=a[1], pattern=t[2], cats=t[3])
          for a in AREAS for t in a[2]]
