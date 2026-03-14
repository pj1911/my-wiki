## Introduction to RAG

RAG stands for Retrieval-Augmented Generation. The name tells us almost everything important:

- Retrieval: before answering, the system searches for relevant external information.
- Augmented: that retrieved information is added to the model's input.
- Generation: the language model then writes an answer using the question plus the retrieved evidence.

A standard large language model (LLM) answers mainly from what it compressed into its parameters during training. That is powerful, but it creates practical problems. The model's knowledge can be stale. It may not know our private or domain-specific documents. It may also produce confident answers without clearly indicating the source of its information. Retrieval-augmented generation (RAG) addresses this limitation by combining a large language model with an information retrieval system:

$$
\text{Answer} = \text{LLM reasoning/generation} + \text{retrieved external evidence}
$$

The LLM still provides language understanding, reasoning, and fluent writing. The retriever provides the evidence. That split is one reason RAG is so attractive: it lets the model reason with current documents rather than only with frozen training memory [[1](#ref-1), [2](#ref-2)].

The easiest intuition for RAG is an exam analogy:

- A plain LLM is closer to a closed-book exam: it answers from memory.
- A RAG system is closer to an open-book exam: it first looks at the relevant pages, then answers.

This does not mean RAG magically makes the model truthful. If it retrieves the wrong pages, or if it misreads the right pages, the answer can still be wrong. But it usually gives the system a better chance to be current, grounded, and auditable [[1](#ref-1), [3](#ref-3)].

**Why did RAG become important?** RAG became important because retraining or fine-tuning a model every time the knowledge base changes is often expensive, slow, or operationally inconvenient. RAG offers a different idea: keep the model mostly fixed, but make the external evidence changeable [[1](#ref-1), [4](#ref-4)].

**Parametric memory versus non-parametric memory.** A natural next question is where the relevant information for answering a query is actually stored. To make this precise, note that a standard LLM already stores information in its trained weights and biases, this is called parametric memory. RAG does not replace that mechanism. Instead, it adds access to non-parametric memory, meaning information stored outside the model. For example, in documents, tables, databases, or vector indices, that can be retrieved at inference time. In this sense, RAG is usually best understood as a system that combines an LLM's parametric memory with external retrieved evidence.

**RAG is not the same as prompting or fine-tuning.** Several ideas closely related to LLM are often discussed together, even though they refer to different things. The following table distinguishes them more clearly.

| Approach | What changes? | Best when | Main weakness |
| --- | --- | --- | --- |
| Prompting | Only the input prompt changes | The model already knows the answer and just needs better instruction | The knowledge may still be stale or unsupported |
| Fine-tuning | The model parameters change | Stable domain behavior should be learned once and reused many times | Updating factual knowledge can be costly and slow |
| RAG | External evidence is fetched at inference time | Answers must be grounded in current, private, or domain-specific sources | Quality depends heavily on retrieval quality |

## RAG glossary

Before going deeper, it helps to define the words that appear over and over in the scientific literature related to RAG.

| Term | Simple meaning | Why it matters |
| --- | --- | --- |
| LLM | A large language model that generates text | It is the generator side of a RAG system |
| Knowledge base / corpus | The collection of documents the system can search | If this is weak, outdated, or messy, RAG will suffer |
| Chunk | A small piece of a larger document | Retrieval usually happens over chunks, not whole books or whole PDFs |
| Embedding | A numerical vector representing meaning | Dense retrieval often compares embeddings instead of exact words |
| Vector index / vector store | A searchable collection of embeddings | This is how many dense retrievers quickly find similar chunks |
| Retriever | The component that finds candidate evidence | It decides what context the generator will see |
| Reranker | A second-stage model that re-orders retrieved results | It can improve quality by filtering noisy results |
| Context window | The amount of text the generator can read at once | Retrieved evidence must fit here, which is why selection matters |
| Grounding / faithfulness | Whether the answer is actually supported by the evidence | A good-looking answer is not enough, it should be justified by sources |
| Provenance / citation | Showing where a claim came from | Important for trust, auditing, and user confidence |
| Query reformulation | Rewriting the user's question for better search | Useful when the original question is vague or hard to retrieve from |
| Hybrid retrieval | Mixing keyword search and semantic search | Often strong in practice because exact terms and semantic meaning both matter |

## How a basic RAG system works

A basic RAG system is usually a pipeline. That means it is not just one model. It is several connected steps.

Knowledge sources → Cleaning and chunking → Indexing \(BM25 / vectors\) → Retrieve top-\(k\) chunks → Rerank / filter → Build prompt with evidence → Generate answer → Citations, checks, logs, feedback

A minimal mathematical picture is:

$$
D = \text{Retrieve}(q), \qquad y = \text{Generate}(q, D).
$$

Here, \(q\) denotes the user's question, \(D\) the set of retrieved documents or chunks, and \(y\) the final answer. This means, first get evidence, then answer from it.

**Step-by-step explanation.**

1. Collect the knowledge source.  
The system first needs an external corpus to retrieve from. This corpus is usually assembled from specific sources relevant to the task, such as a company’s internal documents, product manuals, help-center articles, PDFs, databases, support tickets, code repositories, or selected web pages. In some applications, the source may include internet content, but in many practical RAG systems it is a defined collection of documents rather than the open web.

2. Clean and split documents into chunks.  
Most RAG systems do not retrieve whole documents. They split documents into smaller units called chunks. This matters because the model has limited context space. But chunking is tricky: chunks that are too small lose context, chunks that are too large add noise.

3. Create search representations.  
The system must convert documents into representations that make retrieval possible. One option is sparse retrieval, which represents text through important words and matches documents mainly through lexical overlap, often using methods such as BM25, which scores documents by rewarding important query-term matches. Another option is dense retrieval, which represents queries and passages as embeddings so that retrieval can capture semantic similarity as well as exact wording. Dense Passage Retrieval (DPR) was a major milestone in this direction [[5]](#ref-5).

4. Turn the user question into a search query.  
Sometimes the raw question is used directly. Sometimes it is rewritten into one or more search queries. Better query formulation can drastically improve retrieval.

5. Retrieve the top-\(k\) chunks.  
The retriever returns the most relevant chunks according to its scoring method. The number \(k\) is a design choice: too few chunks can miss needed evidence, too many can bury the signal in noise.

6. Optionally rerank or filter the results.  
Many systems add a reranker or a quality filter because the first-stage retriever is fast but imperfect. Some newer systems go further and evaluate whether retrieval was good enough at all [[6]](#ref-6).

7. Build the final prompt.  
The user question and the retrieved evidence are put together into the prompt for the LLM. The prompt may also instruct the model to cite sources, avoid unsupported claims, or say I do not know if evidence is weak.

8. Generate the answer.  
The LLM reads the retrieved material and writes a response. This is the generation part of RAG.

9. Check, cite, and log.  
Stronger systems may verify the answer, attach citations, log which documents were used, or send the output to human review in high-stakes settings.

## Example Case

User question: According to the 2025 travel policy, can interns book business-class flights?

**What a plain LLM might do:** It may guess from general patterns about company travel policy, but it has no guarantee that its answer matches the organization's actual rules.

**What a RAG system does:**

1. Search the policy documents for chunks related to interns, business class, and air travel.
2. Retrieve the most likely passages.
3. Possibly rerank them so the current version of the policy comes first.
4. Put those chunks into the model prompt.
5. Ask the model to answer only from the retrieved evidence.

**Retrieved evidence:**

- Chunk A: Interns must book economy class for all domestic and international travel unless a medical accommodation is approved.
- Chunk B: Business class is allowed only for employees at director level and above on flights longer than 8 hours.

**Final answer:**
No. Under the 2025 travel policy, interns must book economy class. Business class is restricted to directors and above on long flights, unless a separate medical accommodation applies. [Policy 4.2, 4.7]

This example shows the main value of RAG: it answers from the actual source instead of guessing from generic prior knowledge.

Note that, RAG reduces the risk of hallucination, but it does not guarantee truth. The system can still fail if it retrieves the wrong chunk, chooses an outdated version, or generates claims not fully supported by the evidence [[1](#ref-1), [10](#ref-10), [3](#ref-3)].

## Important design choices in real RAG systems

### Source quality and document preparation

A RAG system inherits both the strengths and the weaknesses of its knowledge source. If the underlying documents are duplicated, outdated, badly parsed, missing metadata, or mixed across versions, retrieval becomes unreliable. In practice, many RAG failures are actually failures of document preparation. For this reason, good practice includes removing duplicates, preserving section titles and document boundaries, keeping version and date metadata, separating trusted from untrusted sources, and ensuring that tables, lists, and headings are not destroyed during parsing.

### Chunking

Chunking may seem simple, but it is one of the most important design choices in RAG: if chunks are too small, relevant information may be split across multiple pieces, while if they are too large, retrieval may include unnecessary text and dilute the most useful evidence.

This is one reason recent work such as RAPTOR explores more structured retrieval for long documents, using recursive summaries and multiple levels of abstraction rather than relying only on flat, fixed-size chunks [[7]](#ref-7). Recent work also suggests that metadata, that is, descriptive information about a document, such as its title, section, date, author, or source, can substantially improve retrieval, especially in repetitive corpora where semantic similarity alone may not be enough to distinguish the most relevant passage [[8]](#ref-8).

### Retriever choice

Retrievers are often grouped into three broad families:

- Sparse retrievers: rely on exact terms, rare words, or token overlap. They are often strong for identifiers, acronyms, codes, and exact names.
- Dense retrievers: use embeddings and semantic similarity. They are strong when wording differs but meaning is similar [[5]](#ref-5).
- Hybrid retrievers: combine sparse and dense signals.

Hybrid retrieval remains popular because it captures both exact lexical clues and semantic similarity. In many practical systems, especially those involving technical vocabulary, product names, or regulations, hybrid retrieval is a strong starting point [[3]](#ref-3).

### Reranking and context filtering

First-stage retrieval is often fast but noisy. A reranker can reorder candidates based on a deeper relevance judgment. Some systems also compress context by keeping only the most relevant sentences or passages.

Corrective RAG (CRAG) goes a step further: it explicitly evaluates retrieval quality and can decide to change the retrieval behavior if the initial evidence looks weak [[6]](#ref-6). This is important because bad retrieval should not always be treated as good enough.

### Prompting and answer policy

The way evidence is presented to the model matters. Strong prompts often do some combination of the following:

- tell the model to answer only from retrieved evidence,
- require citations,
- instruct the model to say when evidence is insufficient,
- separate instructions from retrieved documents clearly,
- discourage unsupported speculation.

Self-RAG pushed this idea further by training a model to decide when retrieval is needed and to critique its own generations with special reflection signals [[9]](#ref-9).

### Evaluation

Evaluating a RAG system is harder than evaluating a plain generator because RAG has multiple moving parts. we may want to measure:

- retrieval quality: did the system fetch the right evidence?
- answer quality: is the final answer correct and useful?
- faithfulness: is the answer supported by the retrieved material?
- citation quality: do the cited sources really justify the claims?
- efficiency: how much latency, memory, and cost were required?

Evaluation surveys and benchmarks emphasize that answer accuracy alone is not enough. A system can answer correctly for the wrong reason, or answer incorrectly even when the correct passage was retrieved [[10](#ref-10), [11](#ref-11), [12](#ref-12)].

| Design choice | Common options | Main tradeoff |
| --- | --- | --- |
| Chunking | Fixed-size, semantic, hierarchical, section-aware | Small chunks improve focus but may lose context, large chunks preserve context but add noise |
| Retriever | Sparse, dense, hybrid | Exact matching versus semantic matching, speed versus depth |
| Reranking | Cross-encoder reranking, heuristic filters, quality scoring | Better precision versus extra latency |
| Metadata usage | Dates, section titles, document ids, source types | Better disambiguation versus more indexing complexity |
| Answer policy | Free-form answer, cite-every-claim, abstain-on-weak-evidence | User friendliness versus strict groundedness |
| Evaluation | Retrieval metrics, answer metrics, faithfulness, cost | A system can optimize one dimension while getting worse on another |

### Common failure modes

Many RAG failures can be grouped into a small number of categories:

1. The relevant information was never indexed.
2. The relevant information was indexed, but retrieval missed it.
3. Retrieval found it, but noisy chunks crowded it out.
4. The model saw the evidence but ignored or misinterpreted it.
5. The evidence itself was outdated, low quality, or contradictory.

This shows that RAG is not just a "prompting trick", its a full system design problem.

## A short timeline of RAG research

| Period | Representative work | Why it mattered |
| --- | --- | --- |
| 2020 | REALM, DPR, RAG | Established retrieval as external memory and showed dense retrieval plus generation could improve knowledge-intensive NLP [[2](#ref-2), [5](#ref-5), [1](#ref-1)]. |
| 2022 | RETRO | Showed that retrieval can scale language modeling itself, not only downstream question answering [[4]](#ref-4). |
| 2023--2024 | Self-RAG, CRAG, RAPTOR | Shifted attention from simple one-shot retrieval toward adaptive retrieval, retrieval correction, and better long-document structure [[9](#ref-9), [6](#ref-6), [7](#ref-7)]. |
| 2024 | CRAG benchmark, RAGBench, evaluation surveys, RAG versus long-context study | Pushed the field toward realistic evaluation, benchmark quality, and cost-aware comparison against long-context alternatives [[12](#ref-12), [11](#ref-11), [10](#ref-10), [13](#ref-13)]. |
| 2025--2026 | SafeRAG, GraphRAG analysis, RAG-Anything, A-RAG, metadata-aware retrieval, utility prediction, A2RAG | Expanded RAG toward security, graphs, multimodal evidence, agentic interfaces, better indexing signals, and cost-aware adaptive reasoning [[14](#ref-14), [15](#ref-15), [16](#ref-16), [17](#ref-17), [8](#ref-8), [18](#ref-18), [19](#ref-19)]. |

## New research directions in RAG

This section answers the question about active research area in RAG. A useful high-level summary is that RAG research has moved from a simple pattern:

*"retrieve a few chunks once, then answer"*

toward a much richer family of systems that ask:

*"should we retrieve, what should we retrieve, how should we organize it, how do we know it helped, and how do we keep the whole system reliable?"*

### Adaptive, self-reflective, and agentic RAG

One major research direction is making RAG adaptive rather than fixed. Self-RAG is important here because it teaches the model to decide when retrieval is needed and to reflect on the quality of its own response [[9]](#ref-9). This matters because not every question needs retrieval, and blind retrieval can sometimes hurt more than help.

Corrective Retrieval-Augmented Generation (CRAG) is another step in this direction. Instead of assuming the retrieved documents are good, CRAG evaluates the quality of retrieval and can change what it does next, including extending retrieval behavior when the initial evidence is weak [[6]](#ref-6).

The broader idea has grown into Agentic RAG (A-RAG): the model behaves more like an agent that plans, decides, searches, reads, and sometimes iterates [[20]](#ref-20). Very recent work such as A-RAG gives the model multiple retrieval tools like keyword search, semantic search, and chunk reading, and lets it adaptively use them [[17]](#ref-17). At the same time, early 2026 comparison work warns that agentic RAG adds cost and complexity, so the right question is not "is agentic better?" but "when is the extra autonomy worth it?" [[21]](#ref-21).

### Long-document retrieval and better retrieval units

A second major direction is improving the unit of retrieval. Traditional RAG often retrieves flat, fixed-size chunks. But long documents do not naturally come pre-divided into ideal retrieval units.

RAPTOR addresses this by building a tree of summaries over the document so retrieval can happen at different levels of abstraction [[7]](#ref-7). This is especially useful for long documents, multi-step reasoning, and questions that require both broad understanding and specific details.

Recent work also argues that metadata matters. In some domains, multiple documents share similar language, so semantics alone may not distinguish them well. Metadata-aware retrieval studies show that section labels, dates, structural cues, and document identifiers can significantly improve disambiguation [[8]](#ref-8).

Another new line of work asks a subtle but important question: will retrieval actually help on this question? Predicting retrieval utility and final answer quality has become its own research topic, because a strong system should ideally know when extra retrieval is useful and when it is unnecessary [[18]](#ref-18).

### Graph-based and multi-hop RAG

Many real questions require connecting facts rather than reading one paragraph. This motivates GraphRAG, where evidence is organized using graph structure, such as entities and relations.

The promise of GraphRAG is intuitive: if a problem involves relationships, a graph may provide cleaner paths for multi-hop reasoning. But recent work has become more skeptical and disciplined. A major 2025 analysis asks when graphs are truly worth using and notes that GraphRAG can underperform vanilla RAG on many real tasks [[15]](#ref-15).

This is an important sign of scientific maturity. Instead of assuming graphs are automatically better, the field is now trying to identify the exact conditions under which graph structure helps. A very recent 2026 paper, A2RAG, pushes this further with adaptive and agentic graph retrieval aimed at reducing cost while remaining reliable in multi-hop settings [[19]](#ref-19).

### Multimodal RAG

Much of the world is not plain text. Real documents often contain images, tables, charts, formulas, scanned layouts, and other forms of mixed text-and-visual evidence.

Traditional text-only RAG ignores much of that signal. This is why multimodal RAG is a major research direction.

A good example is RAG-Anything, which treats multimodal content as interconnected knowledge entities and performs cross-modal hybrid retrieval over text, visual elements, tables, and other structure [[16]](#ref-16). This direction matters because many enterprise and scientific workflows rely on evidence that cannot be faithfully reduced to plain text alone.

### Evaluation, benchmarking, and trust

As RAG matured, researchers realized that benchmark quality matters enormously.

CRAG (Comprehensive RAG) Benchmark was designed to better reflect dynamic and realistic question answering, including web and knowledge-graph style retrieval. Its authors explicitly show that straightforward RAG leaves a large gap to fully trustworthy QA [[12]](#ref-12).

RAGBench provides a large-scale, explainable benchmark tied to industry-style corpora and varied RAG task types [[11]](#ref-11). Evaluation surveys also make an important conceptual point: a RAG system has at least two major subsystems: retrieval and generation, and both need evaluation. A strong answer metric alone is not enough [[10]](#ref-10).

### Security and robustness

A plain beginner intuition is often: if RAG uses external evidence, it should be safer. The reality is more complicated.

External evidence can also become an attack surface. Malicious or manipulated documents may influence retrieval, conflict with legitimate sources, or degrade service quality. SafeRAG formalizes this concern with a security benchmark for RAG systems and shows that external knowledge can make systems vulnerable in new ways [[14]](#ref-14).

So the research frontier is not only about helping models know more. It is also about helping them know safely.

### RAG versus long-context LLMs

As context windows get larger, an important question keeps coming up:

Do we still need RAG if a model can read a very long context directly?

A 2024 study comparing RAG with long-context LLMs found that long-context models can outperform RAG when resourced sufficiently, but RAG still has a strong cost advantage [[13]](#ref-13). This has pushed the field toward hybrid routing: send some questions to long-context reading, others to retrieval, and choose based on the task.

This is a good example of how research has matured. The question is no longer RAG or not? It is increasingly which combination of retrieval, long context, and tool use is optimal for this specific query?

## Current Limitations of RAG

This section turns from the promise of RAG to its limitations, highlighting the main problems that remain unresolved.

**Retrieval quality is still the central bottleneck.** If the wrong evidence is retrieved, the generator may never recover. This sounds obvious, but it remains one of the hardest practical problems because relevance depends on wording, document structure, metadata, domain vocabulary, and user intent. Recent work on corrective retrieval, metadata-aware retrieval, and retrieval-utility prediction all exist because this bottleneck remains unsolved [[6](#ref-6), [8](#ref-8), [18](#ref-18)].

**Correct retrieval does not guarantee faithful generation.** Even when the system retrieves a relevant passage, the LLM may still misread it, overgeneralize from it, combine it incorrectly with prior knowledge, or add unsupported details.

This is why evaluation papers emphasize faithfulness or grounding, not just answer fluency or superficial correctness [[10](#ref-10), [3](#ref-3)]. In other words, RAG can reduce hallucinations, but it does not eliminate them.

**Evaluation is still fragmented.** We still do not have a single universally accepted way to evaluate RAG. Some benchmarks focus on retrieval. Others focus on final answers. Some ignore cost, latency, or citation quality. Others do not reflect realistic multi-document enterprise settings. The recent surge of benchmarks and evaluation surveys is evidence that the problem is still open [[11](#ref-11), [12](#ref-12), [10](#ref-10)].

**Chunking and document structure remain underappreciated.** A lot of RAG quality depends on how documents are split, labeled, and indexed. But chunking is still surprisingly heuristic in many systems. Research such as RAPTOR and metadata-aware retrieval suggests that structure-aware RAG is promising, yet there is still no universally best chunking method across domains [[7](#ref-7), [8](#ref-8)].

**Multi-hop and corpus-level reasoning remain difficult.** Some questions cannot be answered from a single chunk: they require combining evidence across multiple documents or reasoning over relationships between pieces of information. This is one reason GraphRAG and related adaptive graph-based methods have attracted attention. However, this promise should be treated carefully. A detailed 2025 analysis reports that GraphRAG often underperforms vanilla RAG on real tasks, suggesting that the open problem is not simply to build more graphs, but to understand when graph-based retrieval provides a real and measurable advantage [[15](#ref-15), [19](#ref-19)].

**Multimodal and structured evidence are still hard.** Most deployed RAG systems remain text-heavy, while real knowledge work often depends on tables, figures, diagrams, layout, formulas, and images. Multimodal RAG frameworks are emerging, but this area is still early. Cross-modal alignment, retrieval across mixed evidence types, and source attribution in multimodal settings remain open challenges [[16]](#ref-16).

**Security is a first-class problem.** RAG expands the model's access to external knowledge, but that also means it expands the attack surface. If malicious or low-trust sources enter the retrieval space, the system may be manipulated. SafeRAG shows that this is not just a theoretical concern [[14]](#ref-14). Robust source trust, retrieval hardening, and conflict handling remain important open areas.

**Cost and latency trade-offs are still difficult.** Better RAG pipelines often improve performance by adding extra components, such as multiple retrieval stages, rerankers, query reformulation, verification steps, agentic tool use, and long-context fallbacks.

These can improve quality, but they also add latency and cost. Recent comparisons of RAG, long-context LLMs, and agentic RAG show that performance gains must always be judged against compute and engineering complexity [[13](#ref-13), [21](#ref-21), [17](#ref-17)].

**We still do not know perfectly when retrieval is needed.** Some questions need external evidence, others do not. Some need a little retrieval. Others need iterative search or multi-hop evidence. One of the most active modern ideas is that a good RAG system should decide when, how, and how much to retrieve. Self-RAG, A-RAG, and utility prediction papers all point toward this gap [[9](#ref-9), [17](#ref-17), [18](#ref-18)].

**Freshness, provenance, and source conflict are still messy.** Real document collections are rarely neat. They contain outdated versions, contradictory passages, and uncertain authority levels. A useful RAG system should not only find relevant text; it should know which source is authoritative, which version is current, and how to answer when sources disagree. This remains a difficult systems problem [[6](#ref-6), [3](#ref-3)].

**Privacy-preserving and personalized RAG is still immature.** Many attractive use cases involve personal or organizational memory. But that raises privacy and compliance issues. How do we personalize retrieval without leaking sensitive information? How do we maintain auditability when the retrieved memory is user-specific? Recent surveys explicitly point to privacy-preserving retrieval and personalized memory as open directions rather than solved components [[3]](#ref-3).

## Conclusion

RAG is one of the most important ideas in modern LLM systems because it addresses a basic weakness of parametric models: they are powerful reasoners and generators, but their stored knowledge is static, opaque, and not always tied to a source. RAG improves this by letting the model retrieve external evidence at inference time [[1](#ref-1), [2](#ref-2)].

The research frontier, however, is much richer than that simple definition. Current work is moving toward adaptive retrieval, self-reflection, agentic workflows, better long-document handling, graph-structured retrieval, multimodal evidence, stronger evaluation, and security-aware design [[9](#ref-9), [6](#ref-6), [7](#ref-7), [14](#ref-14), [16](#ref-16), [17](#ref-17), [3](#ref-3)]. At the same time, the main gaps remain very real: retrieval quality, faithfulness, evaluation, multi-hop reasoning, multimodality, security, and cost-efficient routing are still open problems [[10](#ref-10), [15](#ref-15), [18](#ref-18)].

## References

1. <a id="ref-1"></a> Patrick Lewis and others. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. 2020. arXiv:2005.11401.

2. <a id="ref-2"></a> Kelvin Guu and others. REALM: Retrieval-Augmented Language Model Pre-Training. Proceedings of the 37th International Conference on Machine Learning. 2020.

3. <a id="ref-3"></a> Chaitanya Sharma. Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers. 2025. arXiv:2506.00054.

4. <a id="ref-4"></a> Sebastian Borgeaud and others. Improving Language Models by Retrieving from Trillions of Tokens. 2022. arXiv:2112.04426.

5. <a id="ref-5"></a> Vladimir Karpukhin and others. Dense Passage Retrieval for Open-Domain Question Answering. 2020. arXiv:2004.04906.

6. <a id="ref-6"></a> Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective Retrieval Augmented Generation. 2024. arXiv:2401.15884.

7. <a id="ref-7"></a> Parth Sarthi and others. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. 2024. arXiv:2401.18059.

8. <a id="ref-8"></a> Raquib Bin Yousuf, Shengzhe Xu, Mandar Sharma, Andrew Neeser, Chris Latimer, and Naren Ramakrishnan. Utilizing Metadata for Better Retrieval-Augmented Generation. 2026. arXiv:2601.11863.

9. <a id="ref-9"></a> Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Self-Reflective Retrieval-Augmented Generation. 2023. OpenReview.

10. <a id="ref-10"></a> Hao Yu and others. Evaluation of Retrieval-Augmented Generation: A Survey. 2024. arXiv:2405.07437.

11. <a id="ref-11"></a> Robert Friel, Masha Belyi, and Atindriyo Sanyal. RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems. 2024. arXiv:2407.11005.

12. <a id="ref-12"></a> Xiao Yang and others. CRAG -- Comprehensive RAG Benchmark. 2024. arXiv:2406.04744.

13. <a id="ref-13"></a> Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, and Michael Bendersky. Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track. 2024.

14. <a id="ref-14"></a> Xun Liang and others. SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model. 2025. arXiv:2501.18636.

15. <a id="ref-15"></a> Zhishang Xiang and others. When to Use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation. 2025. arXiv:2506.05690.

16. <a id="ref-16"></a> Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and Chao Huang. RAG-Anything: All-in-One RAG Framework. 2025. arXiv:2510.12323.

17. <a id="ref-17"></a> Mingxuan Du, Benfeng Xu, Chiwei Zhu, Shaohan Wang, Pengyu Wang, Xiaorui Wang, and Zhendong Mao. A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces. 2026. arXiv:2602.03442.

18. <a id="ref-18"></a> Fangzheng Tian, Debasis Ganguly, and Craig Macdonald. Predicting Retrieval Utility and Answer Quality in Retrieval-Augmented Generation. 2026. arXiv:2601.14546.

19. <a id="ref-19"></a> Jiate Liu and others. A2RAG: Adaptive Agentic Graph Retrieval for Cost-Aware and Reliable Reasoning. 2026. arXiv:2601.21162.

20. <a id="ref-20"></a> Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG. 2025. arXiv:2501.09136.

21. <a id="ref-21"></a> Pietro Ferrazzi, Milica Cvjeticanin, Alessio Piraccini, and Davide Giannuzzi. Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches. 2026. arXiv:2601.07711.
