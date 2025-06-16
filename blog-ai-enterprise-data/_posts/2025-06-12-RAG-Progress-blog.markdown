---
layout: post-wide
title:  "Progress of retrieval-augmented generation (RAG) as of June 2025"
date:  2025-06-12 10:42:32 +0800
category: AI 
author: Hank Li
---

## 1. Introduction to Retrieval-Enhanced Generation (RAG)

### 1.1 Definition of RAG: Core Concepts, Goals and Advantages

Retrieval-Augmented Generation (RAG) is a technical paradigm that aims to enhance the capabilities of LLMs by integrating external knowledge bases. Its goal is to enable LLM to generate more accurate, updated, and verifiable text content, given that the knowledge of traditional LLM is limited by its training data and context window length, resulting in a knowledge cut-off problem or even completely wrong content when faced with queries that are outside its knowledge scope.  RAG "anchors" the model's answers to reliable external facts by dynamically retrieving relevant information from external data sources and providing this information as context to the LLM. 

The key advantages of traditional RAG are reflected in several aspects:

- **Access to domain-specific knowledge**: The RAG system is able to connect to real-time updated databases or domain-specific knowledge bases, enabling the LLM to obtain and utilize the latest information or expertise not included in its pre-training data.
- **Reducing Hallucinations**: By linking generated content to retrieved external facts, RAG significantly reduces the risk of LLM producing inaccurate or fabricated information.
- **Improve user trust** : RAG systems usually provide the source of the information on which their answers are based, which enhances transparency and verifiability, thereby improving users' trust in the system output.
- **Cost-effectivewness** : RAG provides a more cost-effective way to extend the knowledge boundaries of a model compared to retraining or large-scale fine-tuning the entire LLM to acquire new knowledge or adapt it to a specific domain.

At a deeper level, RAG can be viewed as a non-parametric learning method that achieves instant enhancement of model capabilities by establishing a connection between LLM and an external database without modifying the parameters of the model itself. This dynamic knowledge injection mechanism makes RAG not only a technical optimization, but also a paradigmatic innovation that drives LLM from a general-purpose tool to a specialized, high-reliability information system. The combination of LLM's own powerful parametric knowledge and the dynamic, non-parametric external knowledge provided by RAG enables the model to handle a wider range of more complex knowledge-intensive tasks and generate responses that are more context-aware and fact-consistent.

### 1.2 Is RAG (Retrieval-Augmented Generation) Still Relevant when context windows reach 1 million?

As of June 2025, leading frontier models support context windows of 1 million tokens, allowing them to process entire books, large codebases, or hours of video/audio transcripts in a single prompt.

Key Implications of 1M+ Token Contexts:
- Document Processing: ~1 million tokens ≈ 1,500 pages of text (assuming ~600 words/page).
- Multimodal Expansion: Some models now handle text + images/videos within the same long-context window.
- Code & Data Analysis: Full repository analysis (e.g., 100k+ lines of code) without chunking.

Is RAG (Retrieval-Augmented Generation) Still Relevant?

Yes, but its role is evolving:

✅ Still Useful For:
- Dynamic Knowledge: When real-time data (e.g., live APIs, updated databases) is needed.
- Cost Efficiency: Retrieving only relevant snippets can be cheaper than processing 1M tokens.
- Precision: For very large corpora (e.g., entire company document stores), retrieval can outperform brute-force long-context search.

❌ Less Critical For:
- Single-Document QA (e.g., analyzing a book) – native long-context models now suffice.
- Tasks Where Full Context Helps (e.g., complex narrative reasoning).

### 1.3 RAG Basic Process

A typical RAG system can be roughly divided into two core stages: **retrieval** and **generation**. These two stages work closely together to complete the entire process from user query to final response.

The main task of the retrieval stage is to accurately and efficiently find the most relevant information fragments to the user's query from a large-scale knowledge base. This stage usually includes:
- Preprocessing : Cleaning, format conversion, and segmentation of raw data to facilitate subsequent indexing and retrieval.
- Retrieval : Based on user queries, different retrieval algorithms (such as sparse retrieval and dense retrieval) are used to search in the indexed knowledge base.
- Reranking : Rerank the initially retrieved information fragments to increase the probability that the most relevant information will be selected first.
- Pruning : Remove irrelevant or redundant information to ensure that the context passed to the generation stage is concise and effective.

<div class="mermaid">
flowchart LR
    A[User Query] --> B_Subgraph
    
    subgraph B_Subgraph["Retrieval Stage"]
        direction TB
        B1[Preprocessing<br><sub>Cleaning, Formatting, Segmentation</sub>]
        B2[Retrieval<br><sub>Sparse / Dense Retrieval</sub>]
        B3[Reranking<br><sub>Prioritize relevant fragments</sub>]
        B4[Pruning<br><sub>Remove irrelevant info</sub>]
        
        B1 --> B2
        B2 --> B3
        B2 --> B4
    end
    
    B3 --> C[Generation Stage]
    B4 --> C
</div>

The generation phase focuses on using the retrieved information to guide the LLM to generate the final answer. The key components of this phase include:
- Retrieval Planning : Deciding when and how to retrieve, especially when dealing with complex queries or requiring multiple rounds of interaction.
- Multi-source Knowledge Integration : When information comes from multiple documents or data sources, it is necessary to effectively integrate this information and resolve potential conflicts or inconsistencies.
- Logical Reasoning : Reasoning based on the retrieved information and the user's query to generate logical and coherent answers.

<div class="mermaid">
flowchart LR
    subgraph C_Subgraph["Generation Stage"]
        direction LR
        C1[Retrieval Planning<br><sub>When & how to retrieve</sub>]
        C2[Multi-source Integration<br><sub>Merge info from multiple docs</sub>]
        C3[Logical Reasoning<br><sub>Generate coherent answer</sub>]
        C1 --> C2
        C2 --> C3
    end
    
    C3 --> D[Final response]
</div>

In addition to these two core stages, the RAG system also includes a series of interrelated upstream and downstream elements, such as document chunking, embedding generation, and mechanisms to ensure system security and credibility. These elements together constitute the complete operation process of the RAG system, ensuring that the system can respond to user needs efficiently and accurately.

## 2. Naive RAG
Naive RAG constitutes the starting point of the RAG technical route. It combines information retrieval with language model generation, laying the conceptual foundation for subsequent more advanced RAG architectures.


### 2.1 Architecture Blueprint: Indexing, Retrieval, and Generation
Naive RAG mainly includes three core stages: Indexing, Retrieval and Generation.
1. **Indexing phase**: 
&nbsp;  
- *Document Loading* : First, the system loads raw data from various configured sources such as local file directories, databases, APIs, etc. This data can be in a variety of formats, such as PDF documents, Markdown files, Word documents, web page content, and even images.
&nbsp;  
- *Document Transformation* - Chunking/Splitting : After loading, documents, especially long documents, usually need to be divided into smaller, more manageable units, namely "chunks" or "splits". This is mainly to adapt to the context window limitation of LLM and improve the accuracy of retrieval results. Choosing a suitable chunking strategy is crucial, because too small chunks may lose global context, while too large chunks may contain too much irrelevant information, affecting retrieval efficiency and generation quality.
 &nbsp;  
- *Embedding and Storing Vectors* : Each text block is then converted into a numerical vector representation, called "embedding". This process is usually done by a pre-trained embedding model (such as Sentence-BERT, OpenAI Ada, BAAI/bge-m3), which can capture the semantic information of the text block. The generated document embeddings are stored in a database optimized for efficient similarity search, namely a "vector database", such as FAISS, Pinecone, ChromaDB, Postgresql vector etc.  
 &nbsp;  

2. **Retrieval phase**: when a user asks a query, this stage is responsible for finding the most relevant chunks of documents from the vector database.
- *Query Embedding* : The user’s original query (usually a question or an instruction) is converted into a query vector using the same embedding model as document embedding.  
 &nbsp;  
- *Similarity Search* : The system compares the query vector with all document chunk vectors stored in the vector database. The comparison is usually based on some similarity measure, such as cosine similarity or dot product. The retrieval system returns the top K document chunks that are most semantically similar to the query vector.  
 &nbsp;  

3. **Generation phase**: this phase uses the retrieved information to guide the LLM to generate the final answer.
- *Prompt Augmentation* : The most relevant document chunks retrieved are combined with the user’s original query (or prompt) to form an augmented prompt. These document chunks provide additional contextual information for LLM.  
 &nbsp;  
- *LLM Response Generation* : The augmented prompts are fed into a Large Language Model (LLM). LLM uses its powerful language understanding and generation capabilities to generate responses to user queries based on the provided contextual information.
 &nbsp;  

### 2.2 Challenges of Naive RAG
- **Low Precision and Low Recall** : The retrieval process may not be precise enough, resulting in the retrieved document chunks not being fully aligned with the actual requirements of the query (low precision) or failing to retrieve all relevant document chunks (low recall). This usually stems from improper chunking strategy, embedding quality, or similarity search threshold settings.  

- **Risk of outdated or irrelevant information** : If the information in the knowledge base is not updated in a timely manner, or the retriever fails to accurately identify relevant content, the LLM may be provided with outdated or irrelevant information, resulting in the generation of incorrect answers or hallucinations.  

- **Redundancy and repetition** : When multiple retrieved document blocks contain similar or repeated information, the enhanced hints may appear redundant and the generated results of LLM may also appear unnecessary repetition.  

- **Context integration problem** : When multiple related paragraphs are retrieved, how to effectively sort them and how to coordinate the different styles or tones that may exist among them is a challenge for naive RAG.  

- **Over-reliance on enhanced information** : Sometimes LLMs may over-rely on the retrieved context, merely reciting its contents rather than engaging in deeper synthesis, reasoning, or creative generation.  

- **Loss of contextual information** : Overly simple chunking strategies (such as fixed-size chunking) may cut important contextual connections at chunk boundaries, resulting in incomplete information within a single chunk, thereby sacrificing key contextual information and significantly impairing retrieval accuracy and context understanding capabilities.

## 3. Advanced RAG
To overcome the limitations of naive RAG, researchers have proposed a series of advanced RAG strategies, which aim to comprehensively improve the performance, robustness, and efficiency of the system by optimizing each stage of the RAG process, namely pre-retrieval, retrieval, and post-retrieval.  

### 3.1 Pre-retrieval enhancement
The goal of pre-retrieval enhancement is to improve the quality and structure of the indexed data so that the retrieval process can obtain more accurate and comprehensive information related to the query. Garbage in, garbage out. If the knowledge based itself is not fully prepared, the retrieval and generation, no matter how advanced, will hardly make up for the defects of the source.

#### 3.1.1 Advanced Chunking Strategy
Chunking is to divide the original document into small segments to adopt the model context window limitations. Naive RAG usually use fixed-size chunking which can easily undermine the semantic integrity of the text. Advanced RAG aims to adopt a smarter chunking strategy to preserve the semantic information and contextual connections of the original document.  

Choosing the optimal chunking size and strategy is critical, as it directly affects the accuracy and recal of retrieval. The chunking strategy, however, depends on specific task and model used.  

- **Semantic Chunking** : it segments text based on its natural semantic boundaries. It can divide text chunks based on sentence structure, paragraphs, or topic change points. This approach helps maintain the coherence and integrity of the information within each chunk.  

- **Hierachical Chunking** : it organize long documents with complex structures into a nested hierachical structure, for example, dividing documents into chapters, which are further divided into paragraphs, and paragraphs into sentences. This structure helps capture the overall theme of the document and drills down into specific details, providing support for queries of different granularities.

- **Dynamic granularity** :[the paper link](https://arxiv.org/abs/2406.00456 "Mix-of-Granularity: Optimize the Chunking Granularity for Retrieval-Augmented Generation")
  - **Mix-of-Granularity (MoG)** : Recognizing that different types of queries may require information chunks of different granularities, the MoG approach dynamically adjust the granularity of knowledge base retrieval. It utilizes a trained "router" that draws on the idea of the Mix-of-Experts architecture in machine learning to dynamically select or combine reference fragments of different granularity levels based on the characteristics of the input query. The router is trained through supervised learning, enable it to determine the ideal chunk size for each query.  

  - **Mix-of-Granularity Graph (MoGG)** : The core idea of MoGG is to build the reference document (or the entire knowledge base) into a graph structure in the preprocessing stage. In this graph, related text fragments, even if they are physically far apart in the original document, can be connected by the edges of the graph and become neighbor nodes. This graph structure enables the system to retrieve scattered information more efficiently and supports tasks that require multi-hop reasoning. In addition, in order to solve the problem of blocked gradient backpropagation when training such retrieval models containing top-k selection, MoGG introduces a loss function using "soft labels". These soft labels can be generated by offline algorithms or models such as TF-IDF or RoBERTa as approximate training signals, thereby avoiding hard top-k selection during training, allowing gradients to propagate smoothly, accelerating the training process, and improving the recall and relevance of the model.


#### 3.1.2 Using Metadata to Enhance Indexing and Filtering
Metadata is data that describes data, such as the author, creation date, category, source, keywords, etc. In the RAG system, adding metadata to document blocks and storing them together with vector embeddings during indexing can enhance the flexibility and accuracy of retrieval.


- **Metadata Types** : 
- System metadata: automatically generated by the system such as chunk_id, filename, source, etc.  

- user-defined metadata: descriptive inforamtion manually provided by users based on business needs, such as product_area, genre, document_type.  

- Automatic metadata: structured info automatically extracted from documents by models such as [Vectorize's Iris model](https://vectorize.io/). 


- **Application of metadata** : 

- **Filtering the search space** : Metadata allows for attribute filtering based on semantic search. The system can first filter out documents that match the document type through metadata, and then perform semantic matching among these documents.  

- **Improve search relevance and ranking** : Metadata can be used as an additional basis for sorting. For example, when searching for news articles, the most recently published articles can be displayed first; when searching for internal company documents, the documents can be sorted according to their "officiality" or "update frequency".

- **Providing context to the LLM** : Metadata itself can also be provided as contextual information to the LLM to help it better understand the retrieved content. For example, telling the LLM whether a certain text comes from a "user manual" or a "forum discussion" can affect how the LLM interprets and uses the information.

#### 3.1.3 Knowledge Graph (e.g., GraphRAG)
Knowledge Graphs (KGs) provide a more powerful way of knowledge representation and retrieval for RAG by representing information as a network structure of entities (nodes) and relationships (edges).

- **GraphRAG’s core innovation** : The GraphRAG paradigm addresses the limitations of traditional RAG through the following key innovations:
  - **Graph-structured knowledge representation** : Explicitly captures the relationships between entities and the hierarchical structure within the domain, making the organization of knowledge more consistent with human cognition and more conducive to complex machine reasoning.

  - **Efficient graph-based retrieval technology** : It can achieve context-preserving knowledge retrieval and support multi-hop reasoning. This means that the system can jump from one entity to another related entity along the relationship chain in the graph, thereby collecting information distributed in different knowledge fragments required to answer complex questions.

GraphRAG is mainly aimed at solving the challenges faced by traditional RAG in the following aspects:
- Understanding of complex queries in professional fields.
- Integration of knowledge across distributed sources.
- System efficiency bottlenecks in large-scale applications. By leveraging the structural advantages of knowledge graphs, GraphRAG can better understand query intent, more effectively integrate scattered knowledge, and retrieve in a more efficient way.

It is worth noting that enterprise data consists of a large volume of ralational database, GraphRAG is not designed to query. How to integrate ChatBI such as text2SQL and text2Python with RAG will be critical for the large adoption of RAG in enterprise data analysis landscape. Synlian Data@Source aims to build an integrated platform called SynGraph to integrate ChatBI and RAG for intelligent enterprise LLM applications, leveraging the power of both RAG and enterprise data analysis. 

#### 3.1.4 Optimizing index structure and alignment
Advanced RAG also focuses on optimizing the index structure itself and ensuring better alignment between the index data and potential queries. This may include:

- **Refinement of corpus preprocessing techniques**: For example, for financial documents, can use Markdown to restructure documents to preserve their inherent structure, or enhance table table such as adding row and column text annotation to improve contextual understanding.  

- **Adjustment of indexing strategy**: select or design a more approprite indexing strategy based on data characteristics and expected query types such as differentiate scenarios that require exact maching and scenarios that require semantic matching. 

Investment and optimization in the pre-retrieval stage is one of the key factors that determine whether the RAG system can ultimately succeed in complex real-world scenarios. This also reflects an increasingly obvious trend in the development of RAG technology: data preparation itself is becoming more and more "intelligent."

### 3.2 Optimization of Retrieval
The retrieval process aims to quickly and accurately find the most relevant information from the knowledge base based on the user's query. Advanced RAG strategies introduce a variety of innovations in this process, aiming to improve the accuracy, completeness, and relevance of retrieval.

#### 3.2.1 Core retrieval methodologies
- **Sparse Retrieval** is mainly based on keyword matching, such as the classic [BM25 algorithm](https://en.wikipedia.org/wiki/Okapi_BM25). They evaluate relevance by calculating the overlap between the terms in the query and the terms in the document (usually considering term frequency and inverse document frequency). The advantage of sparse retrieval is its high computational efficiency, which is very effective for scenarios where there is a direct lexical correspondence between the query and the document.  

- **Dense Retrieval**: Dense retrieval uses a deep learning model (usually a Transformer-architecture encoder such as DPR [Karpukhin et al., 2020](https://arxiv.org/pdf/2004.04906) or Contriever [Izacard et al., 2021](https://arxiv.org/pdf/2112.09118)) to encode both the query and the document into a low-dimensional, dense vector space. The relevance is then measured by calculating the similarity between these vectors (such as cosine similarity). The core advantage of dense retrieval is that it can capture semantic similarity. Even if there are no shared keywords between the query and the document, as long as they are similar in meaning, they can be matched. This helps overcome the problem of vocabulary mismatch.  

- **Hybrid Retrieval** : Hybrid retrieval is a method that combines the advantages of sparse retrieval and dense retrieval. It usually combines the scores of sparse retrieval (such as BM25 score) and dense retrieval (such as vector similarity) in some way (such as weighted summation) to obtain more robust retrieval performance than a single method.  

  - **Dynamic Alpha Tuning (DAT)** is a novel hybrid search optimization technology. Traditional hybrid search usually uses a fixed weight factor (alpha) to balance the contribution of sparse and dense searches, and this factor often requires offline tuning. DAT believes that the optimal weight should be dynamically adjusted according to the characteristics of each specific query. It uses an LLM to evaluate the effectiveness of the Top 1 results returned by sparse and dense searches, and assigns an effectiveness score to each method. Then, the alpha value of the current query is dynamically calculated based on these scores, so as to more intelligently balance the contribution of the two search methods.   

  - **Graph Retrieval** : When the knowledge base is organized in knowledge graph, graph retrieval techniques can be used. Such techniques consider not only the content of the nodes (entities) themselves, but also the relationships (edges) between nodes, thereby enabling more complex path- or subgraph-based retrieval, especially for queries that require multi-hop reasoning.

#### 3.2.2 Query Augmentation
User queries are often ambiguous, incomplete, or inconsistent with the expression of the documents in the knowledge base. Query enhancement technology aims to bridge this semantic gap by modifying or expanding the original user query.

- **Query Expansion** : add relevant terms, concepts, or context to the original user query so that it can match a wider range of relevant documents.

  - **LLM-QE (LLM-based Query Expansion)** uses LLM to generate "document-like" extensions for the original query. For example, given a short query, LLM generates a more detailed text that may contain the answer to the query. A core innovation of LLM-QE lies in its training method: it designs rank-based rewards (evaluating the similarity between the extension content and the real relevant documents) and answer-based rewards (evaluating the relevance of the extension content to the answer generated by LLM based on the real document), and uses the Direct Preference Optimization (DPO) algorithm to fine-tune LLM so that the extension content it generates is more in line with the preferences of the retriever and LLM itself, while reducing the hallucinations that LLM may produce during the expansion process. [the paper link](https://arxiv.org/html/2502.17057v1). It is worth noting that [LLM-based Query Expansion Fails for Unfamiliar and Ambiguous Queries](https://arxiv.org/pdf/2505.12694).  

- **Query Rewriting/Decomposition** : For complex or multifaceted queries, direct retrieval may not be effective. Query rewriting or decomposition techniques aim to transform the original complex query into a clearer and easier to retrieve form, or decompose it into multiple simpler sub-queries.

  - [LevelRAG](https://github.com/ictnlp/LevelRAG) adopts a hierarchical search strategy. It first uses a "high-level searcher" to decompose the user's original query into multiple atomic sub-queries. These sub-queries are then distributed to different "low-level searchers", which can be sparse retrievers, dense retrievers, or network searchers. In each low-level searcher, LevelRAG also uses LLM to rewrite or refine the atomic query assigned to it to make it more suitable for the characteristics of the corresponding retriever. Finally, the high-level searcher aggregates the retrieval results from all low-level searchers to generate the final answer. An important feature of this approach is that it separates retrieval logic (such as multi-hop planning) from retriever-specific query rewriting, providing greater flexibility.  

  - **MA-RAG (Multi-Agent RAG)** uses multi-agent collaboration to handle complex queries. It includes agents with different roles, such as planner, step definer, extractor, and QA Agents, which work together to handle subtasks such as query disambiguation, evidence extraction, and answer synthesis. [see the paper](https://arxiv.org/pdf/2505.20096)

- **Iterative Query Refinement** dynamically adjust or generate new queries based on the results of the first round of retrieval or in the continuous reasoning process to gradually approach the required information.  

LLM's powerful natural language understanding and generation capabilities can be used to optimize the retrieval process itself, especially in dealing with the ambiguity and complexity of user queries. This has led to the emergence of technologies such as LLM-QE that use LLM for query expansion, and LevelRAG and MA-RAG that use LLM for query decomposition, rewriting or planning. Even in hybrid retrieval strategies such as DAT, LLM is used as a "referee" to evaluate the effectiveness of different retrieval methods. This tighter coupling and smarter interaction between LLM and retrieval modules indicates that RAG systems are evolving from a simple "retrieval-generation" pipeline to a more dynamic and adaptive information acquisition and reasoning system. 

### 3.3 Post-retrieval Refinement
The goal of the Post-Retrieval Refinement stage is to optimize these retrieved contexts to better fit the context window constraints of the LLM, reduce noise interference, and ensure that the most relevant and important information can be effectively utilized by the LLM.

### 3.3.1. Advanced Re-ranking Strategy
The list of document chunks returned by the initial search may still contain some less relevant or redundant content, or its ordering may not be optimal. Reranking aims to refine this list through a more detailed evaluation, placing the most useful and relevant pieces of information in a more prominent position, thereby improving the quality of the generated answers.

- **Common methods** include re-scoring document chunks based on a more sophisticated semantic similarity calculation (possibly using a more powerful model than the initial retrieval, such as a cross-encoder); or adjusting the ranking based on heuristic rules, for example, placing document chunks containing entities or keywords directly related to the query higher. Other studies have shown that placing the most relevant context at the beginning or end of the prompt (i.e., "marginal positions") may help LLMs make better use of this information, as LLMs sometimes suffer from the problem of "lost in the middle", that is, paying less attention to information in the middle of a long context.  

- **RS Score (Relevancy Score) for Multimodal RAG** : In RAG systems that process multimodal information (such as text-image pairs), RS Score is proposed as a measure to quantify the relevance between queries and retrieval items. It is usually a scalar score between 0 and 1, with higher scores indicating stronger relevance. RS Score models usually use a specifically fine-tuned visual language model (VLM) to learn the semantic relationship between queries and retrieval items (such as images or text documents). By training on a dataset containing positive and negative sample pairs, the RS Score model aims to distinguish relevant and irrelevant data more accurately than traditional CLIP embedding cosine similarity-based methods, especially in judging irrelevance.  

- **METEORA (Rationale-driven Selection)** : [METEORA](https://arxiv.org/pdf/2505.16014) proposes an innovative approach to replace traditional re-ranking with "rationale-driven selection". The idea is to first use a preference-tuned LLM (using direct preference optimization DPO technology) to generate a series of "rationales" for the input query, that is, phrases or sentences explaining why certain specific information is needed. Then, an "Evidence Chunk Selection Engine" (ECSE) uses these rationales to guide the selection process of evidence chunks. ECSE works in three stages:  

  - **Local relevance pairing** : Pair each reason with the retrieved evidence chunk and evaluate the local relevance.  
  
  - **Global selection based on inflection point detection** : An adaptive cutoff point is determined through a global selection mechanism (such as an inflection point detection algorithm) to select the most important set of evidence blocks, avoiding the rigid setting of the K value in top-K in traditional methods. 
  
  - **Contextual expansion through neighboring evidence**: Context adjacent to the selected evidence block may be included to ensure the integrity of the information. In addition, METEORA uses a "Verifier LLM" to check the consistency between the selected evidence and the reasons to detect and filter possible "poisoning" or misleading content. Since the reasons are used consistently in both the selection and verification process, METEORA provides an explainable and traceable evidence flow, improving the robustness and transparency of the system.

#### 3.3.2. Context Compression Technology
The context window length of LLM is limited, and even the latest models cannot process input information infinitely. When the total length of the retrieved relevant document block exceeds the context window limit of LLM, the context needs to be compressed to reduce its total volume while retaining the core information.

- **Hard Compression** directly modifies the surface structure of the text, such as removing unimportant sentences or paragraphs through pruning, or generating shorter text representations through summarization. These methods are usually easy to understand and implement, but the compression rate is limited and some detailed information may be lost.  

- **Soft Compression** aims to compress document content into a more compact vector representation, or generate attention key-value pairs for the model to refer to during generation. This type of method may sacrifice a certain degree of interpretability in exchange for higher compression rate and efficiency.

- **MacRAG (Multi-scale Adaptive Context RAG)** is a hierarchical retrieval framework that processes documents at multiple scales offline. First, it segments the document into partially overlapping blocks, and then compresses these blocks through methods such as abstract summarization. These compressed fragments are further divided into finer-grained units for building hierarchical indexes. At query time, MacRAG first retrieves the most precise fragments from the finest level, and then gradually expands upwards, merging adjacent blocks, parent blocks, and even document-level expansion, thereby dynamically building an effective long context for a specific query that covers enough information and has controlled length. [see the paper](https://arxiv.org/pdf/2505.06569)  

- **PISCO (Pretty Simple Compression)** claims to achieve up to 16x compression rates on RAG tasks with minimal accuracy loss (0-3%). A key feature of PISCO is that it does not rely on pre-training or annotated data, but instead trains the compression model entirely through sequence-level knowledge distillation from document-based questions. This means that the compression model (student model) is trained by learning the output generated from the teacher model (using uncompressed context). PISCO's compressor encodes each document into a fixed set of "memory embeddings", and the decoder generates answers based on the query and these compressed embeddings. [see the paper](https://arxiv.org/pdf/2501.16075) 

Various techniques in the post-retrieval refinement stage, especially re-ranking and context compression, are crucial to solving the "middle content loss" problem that may occur when LLM processes long contexts. It is not enough to simply retrieve a large number of relevant documents; how to effectively filter, sort, and compress this information and present it to LLM in the best way is directly related to the quality of the final generated answer. This makes post-retrieval processing an indispensable optimization link in the RAG process, and its goal is to make the retrieved context more "digestible" and "easy to use" for LLM.

## 4. Specific RAG architecture
In addition to general optimization at various stages, a series of RAG architectures with unique design concepts and specific application goals have emerged. These architectures represent the deepening and specialization of RAG technology in different directions, aiming to solve more complex problems or achieve more advanced functions.

### 4.1. Iterative and Multi-Hop RAG Framework
Many complex queries cannot be effectively answered through a single round of retrieval and generation. They often require evidence from multiple information sources or a series of reasoning steps to reach a conclusion. Iterative and multi-hop RAG frameworks are designed to address such challenges. Their core idea is to gradually build and improve the knowledge required to answer questions through multiple retrieval-reasoning cycles.

-  **Self-RAG**: This is a self-reflective retrieval-augmented generation framework. Self-RAG is unique in that it trains a single LLM to adaptively retrieve passages on demand (it can retrieve multiple times or skip retrieval entirely), and uses special "reflection tokens" to generate and evaluate the retrieved passages as well as the model's own generated content. These reflection tokens enable the model to evaluate the necessity of retrieval (whether retrieval is needed), the relevance of the retrieved content (whether the retrieved content is useful), and whether the generated content is supported by evidence, complete, etc. The training process involves a retriever, a critic, and a generator. At inference time, Self-RAG can use these reflection tokens for tree-decoding to choose the best generation path based on preferences for different evaluation dimensions (such as evidence support, completeness). [see the github](https://github.com/AkariAsai/self-rag).    

- **Auto-RAG**: Auto-RAG emphasizes autonomous iterative retrieval. In this framework, LLM conducts multiple rounds of dialogue with the retriever to systematically plan retrieval steps and optimize queries to obtain valuable knowledge. This process continues until enough information is collected to answer the user's question. The training of Auto-RAG relies on an autonomously synthesized, reasoning-based decision instruction that enables it to autonomously adjust the number of iterations based on the difficulty of the question and the utility of the retrieved knowledge without human intervention.  [see the github](https://github.com/Marker-Inc-Korea/AutoRAG)  

- **KnowTrace** : KnowTrace rephrases the iterative RAG process as a process of knowledge graph expansion. Rather than simply stacking retrieved text fragments, it allows LLM to actively track and complete knowledge triples (subject-predicate-object) related to the question, thereby dynamically building a knowledge graph specific to the current question. This gradually constructed knowledge graph provides LLM with a structured and evolving context, which helps LLM to make clearer and more organized reasoning, while also effectively filtering out redundant or misleading information. [see the github](https://github.com/rui9812/KnowTrace)

- **RAG-Fusion** : RAG-Fusion aims to enhance the comprehensiveness of retrieval by generating multiple query perspectives. It first lets LLM generate multiple relevant sub-queries based on the original user query. Then, it performs vector search on each sub-query separately to obtain relevant documents. Next, it uses the Reciprocal Rank Fusion (RRF) algorithm to re-rank and score-fuse all retrieved documents to obtain a comprehensive ranked list. Finally, this fused ranked document list is provided to LLM along with the original query and generated sub-queries to generate the final answer. This approach helps to generate more comprehensive and in-depth answers by exploring the information space from different perspectives.[see the github](https://github.com/Raudaschl/rag-fusion)

- **LevelRAG**[discussed above](#322-query-augmentation): As mentioned earlier (Section 3.2.2), LevelRAG's high-level searcher naturally supports the implementation of multi-hop logic by decomposing complex queries into atomic sub-queries.

### 4.2. Self-Correcting and Reflective RAG Paradigm
In order to improve the reliability and accuracy of RAG systems, researchers have developed RAG paradigms with self-correction and reflection capabilities. These systems have built-in mechanisms to evaluate the quality of retrieved information and generated responses, and can perform error correction or response optimization based on the evaluation results.

- **CRAG (Corrective RAG)** : The core of CRAG lies in its error correction capability. It contains a lightweight retrieval evaluator that evaluates the overall quality of the retrieved documents and gives a confidence score. According to this confidence, the system triggers different knowledge retrieval operations: if the confidence is high (marked as "Correct"), the retrieved document is used directly; if the confidence is low (marked as "Incorrect"), the system may initiate a web search to obtain more accurate or comprehensive information for supplement or replacement; if the result is uncertain (marked as "Ambiguous"), additional retrieval or correction steps may also be triggered. In addition, CRAG also uses a "decompose-then-recompose" algorithm to optimize the retrieved documents, aiming to selectively focus on key information and filter out irrelevant content. [see the github](https://github.com/facebookresearch/CRAG)

- **AlignRAG** focuses on solving the problem of "reasoning misalignment", that is, the possible inconsistency between the internal reasoning process of LLM and the retrieved external evidence. It introduces the "Critique-Driven Alignment" (CDA) mechanism. At its core is a "Critic Language Model" (CLM), which is trained through contrastive learning, can identify misalignments in the reasoning process, and generate structured critical information to guide the alignment process. At test time, AlignRAG treats the generated reasoning process as an optimizable object, and iteratively corrects it through the critical information provided by CLM, thereby transforming the RAG pipeline into an active reasoning system that dynamically aligns the generated content with the retrieved evidence. An important feature of AlignRAG is that it can be integrated into the existing RAG pipeline as a plug-and-play module. [paper](https://arxiv.org/pdf/2504.14858v3) 

### 4.3. Modular RAG Architecture
Modular RAG represents a more flexible and composable RAG system design method. It decomposes the complex process of RAG into a series of independent and configurable modules, such as query processing module, retrieval module, filtering and sorting module, context enhancement module, response generation module, post-processing module, etc.

- **Advantages** : This modular design brings many benefits:
  - **Flexibility and customizability**: Each module can be independently selected, replaced or fine-tuned according to specific application requirements. For example, a specialized retriever or reranker can be customized for a specific domain.
  - **Scalability** : The system can be horizontally expanded by deploying different modules on different resources.
  - **Maintainability** : When a module has a problem or needs to be upgraded, it can be debugged and updated independently without affecting other parts of the entire system.
  - **Promote innovation** : Modular interfaces make it easier to integrate new research results or third-party tools.  

- **Implementation** : This type of architecture is usually built with the help of frameworks such as LangChain and LlamaIndex, which provide rich preset components and flexible orchestration capabilities.  

- **Typical modules** : A complex modular RAG system may include: query preprocessing (such as rewriting, disambiguation), multi-source retrieval, metadata filtering, advanced reranking, context enhancement (such as integrating knowledge graphs, calling APIs to obtain dynamic data), context compression, LLM generation, fact verification, formatted output, as well as user feedback collection and model iteration.

### 4.4. Hybrid Document RAG (HD-RAG)
Enterprise documents often contain multiple types of data, such as plain text, tables, images, etc. Traditional RAG systems are mainly optimized for plain text and face challenges when processing mixed documents containing complex structures (such as hierarchical tables). The HD-RAG (Hybrid Document RAG) framework aims to solve this problem and effectively integrate heterogeneous data such as text and tables to support more comprehensive retrieval and generation. [see paper](https://arxiv.org/pdf/2504.09554)

**HD-RAG core components**:
**Corpus Construction Module**: For tables in mixed documents, especially complex tables with hierarchical structures, HD-RAG adopts a "Hierarchical Row-and-Column-Level" (H-RCL) table summarization method. This method aims to capture the structural information and content of the table and generate a representation that can both preserve the table structure and optimize retrieval.

**Retrieval Module** : To overcome the limitations of single semantic similarity retrieval, HD-RAG adopts a two-stage retrieval strategy. The first stage is ensemble retrieval, which combines BM25 (for keyword matching) and embedding-based semantic retrieval (for semantic understanding) to screen candidate documents from different perspectives. The second stage is LLM-based retrieval, which uses the contextual reasoning ability of LLM to further identify the most relevant documents from the candidate documents.

**QA Inference Module** : This module uses a prompting strategy called RECAP (Restate, Extract, Compute, Answer, Present) to accurately extract and utilize information from mixed documents, supporting multi-step reasoning and complex calculations. The steps of RECAP include: restating the question, extracting relevant data, calculating the answer, answering the question, and (for calculation questions) presenting the calculation formula.

### 4.5. Multi-agent RAG (MA-RAG)
MA-RAG introduces the idea of Multi-Agent System (MAS), which enables a group of AI agents with different expertise to work together to handle various subtasks in the RAG process. [paper](https://arxiv.org/pdf/2505.20096)

**Core concept**: MA-RAG decomposes the complex RAG task into a series of smaller and more manageable subtasks, such as query disambiguation, evidence extraction, answer synthesis, etc., and assigns these subtasks to specialized agents for processing.  

**Agent roles** : A MA-RAG system may include a planner agent responsible for overall task planning, a step definer agent responsible for refining execution steps, an extractor agent responsible for extracting information from documents, a QA agent responsible for generating the final answer, etc. These agents can be dynamically called according to task requirements to form an efficient workflow.

**Advantages**: Through task decomposition and agent collaboration, MA-RAG can better cope with the ambiguity and reasoning challenges inherent in complex information query tasks, improving the robustness of the system and the interpretability of the results.

 After naive RAG and early "advanced RAG" provided a universal method for connecting LLM with external knowledge, researchers and developers began to face more specific and difficult real-world problems, such as how to perform multi-step reasoning in the true sense, how to handle complex documents containing tables and text, how to ensure the reliability of the system in the face of noisy or misleading information, and how to decompose complex RAG processes into manageable and optimizable modular components. It is these specific needs that drive the formation of specific technical routes such as iterative RAG, self-correcting RAG, HD-RAG, modular RAG, and MA-RAG. This specialization enables RAG technology to be more effectively applied to a wider range of more challenging tasks, and also marks that the RAG field is maturing, evolving from a "one-size-fits-all" solution to an advanced technology system with a rich "toolbox" that can flexibly respond to diverse needs.

## 5. Fine-Tuning Strategy in RAG System
In order to further improve the performance of the RAG system, in addition to innovations at the architectural level, fine-tuning the core model components in the RAG system, namely the Retriever and Generator, is also an important technical route. Fine-tuning aims to make these general models more adaptable to the specific task requirements and data characteristics of RAG.

### 5.1. Optimizing the Retriever: Fine-tuning the Embedding Model
The core of the retriever is the embedding model, which is responsible for converting text (query and document blocks) into vector representations. The quality of the embedding model directly determines the relevance of the retrieval results. By fine-tuning the embedding model, the vectors it generates can better capture the semantic nuances in a specific domain or task, thereby improving retrieval accuracy.

- **Domain-adaptive fine-tuning** : A general pre-trained embedding model may not perfectly capture the relationship between specialized terms or concepts in a specific domain. By fine-tuning the embedding model on the corpus of the target domain, it can learn semantic representations that are more in line with the characteristics of the domain. For example, in the financial field, the embedding model can be fine-tuned using financial documents to improve its understanding of financial terms and concepts.
- **Instruction Fine-tuning to Support Multi-task and Domain-specific Retrieval** : An emerging strategy is to adopt instruction fine-tuning to train the retriever encoder to handle multiple retrieval tasks and adapt to the needs of specific domains. This approach usually chooses a small embedding model with large context length and potential multilingual capabilities as the basis. Then, fine-tuning is performed by constructing a training dataset containing a variety of instruction templates and positive and negative sample pairs. For example, instructions can be designed to guide the model to retrieve specific types of steps, tables, fields, or retrieve directory items based on user requirement descriptions. Training data can be extracted from existing internal databases or application training sets without the need for extensive manual annotation. By training with contrastive learning loss, the model can learn to bring semantically similar text-object pairs closer in the embedding space and push dissimilar pairs farther away. This approach aims to achieve a unified retriever that is scalable, fast, and can serve multiple use cases at a low cost.
- **Dynamic Embeddings** : Some studies have explored dynamic embedding techniques, which can better capture the dynamic changes of context rather than just static semantic representations.

### 5.2. Optimizing the Generator: Adapting LLMs to RAG
The generator LLM is responsible for leveraging the retrieved context to generate the final answer. Even if the retrieved context is of high quality, if the LLM is not optimized for RAG scenarios, it may still fail to fully utilize the context, or even produce hallucinations when the context is imperfect (e.g., contains noisy or incompletely relevant information). Fine-tuning the generator aims to enhance its ability to understand and leverage the retrieved context, and improve the factual consistency and relevance of the generated answers.

- **Finetune-RAG** : This method is specifically designed to train LLM to resist hallucination in the face of imperfect retrieval (i.e., the retrieved context contains both correct and fictitious/misleading information). The core idea is to construct a special training dataset where each sample contains a correct document block related to the question and a fictitious document block that may mislead the model. LLM is trained to rely only on the correct document block to generate reference answers, thereby learning to ignore or discern false information. [paper](https://arxiv.org/abs/2505.10792)

- **ALoFTRAG (Automatic Local Fine Tuning of Retrieval Augmented Generation models)**: ALoFTRAG proposes an automated, localized fine-tuning framework that aims to improve the accuracy of RAG systems on domain-specific data without manually annotating data or relying on large teacher models. The framework first automatically generates synthetic training data from unannotated domain-specific texts, including questions, answers, correct reference texts, and "hard negative texts" (texts that are related to the question but are not the source of the correct answer). Then, using this synthetic data, the generator LLM is fine-tuned through LoRA (Low-Rank Adaptation), a parameter-efficient fine-tuning technique. The training goal of ALoFTRAG is to let LLM learn to first accurately cite the correct text source from a provided list of reference texts (including correct and distracting items), and then generate answers based on that source. This approach not only improves the accuracy of the answers, but also improves the accuracy of the citations. [paper](https://arxiv.org/abs/2501.11929)

Fine-tuning the retriever and generator are not two isolated processes, but rather they are mutually dependent and reinforcing. A better performing retriever can provide the generator with more relevant and accurate contextual information. However, even with perfect retrieval results, if the generator is not optimized for RAG scenarios, it may still fail to fully utilize these high-quality contexts or produce inaccurate outputs in the face of even slight contextual imperfections. Conversely, a generator that is carefully fine-tuned, can efficiently utilize context and resist interference (such as models trained by Finetune-RAG or ALoFTRAG), if its upstream retriever performs poorly and continuously provides noisy or irrelevant context, then the advantages of the generator will be difficult to fully exploit.

Therefore, achieving the optimal performance of the RAG system often requires finding a balance between fine-tuning the retriever and the generator, and may even require a strategy of co-evolution or iterative optimization. This means that future RAG system development may increasingly tend to holistic optimization solutions rather than isolated improvements to individual components. This also puts forward new requirements for the construction and fine-tuning methodology of training data, that is, more data sets and training processes designed for end-to-end RAG performance optimization are needed.

## 6. Other perspectives and emerging considerations
As RAG technology continues to evolve, some new perspectives and considerations have begun to emerge, which may have an impact on the mainstream technology route of RAG in the future and even give rise to alternative solutions.

### 6.1. Cache-Enhanced Generation (CAG) as a Potential Alternative
Cache-Augmented Generation (CAG) is a new paradigm proposed for specific scenarios that may replace traditional RAG. Its core idea is that when dealing with a limited and relatively stable knowledge base, all relevant resources can be pre-loaded into the extremely long context window of the Large Language Model (LLM) and the key-value pairs (Key-Value Cache, KV Cache) generated during its runtime can be cached.

- **Working mechanism** :
  - **Preloading and caching** : In the offline phase, the entire knowledge base (or its relevant subset) is provided as input to the LLM with long context capability, and the KV cache generated during the forward propagation process is calculated and stored. This computational cost only needs to be borne once.
  - **Inference** : During actual inference, when a user asks a query, the system loads the pre-computed KV cache and feeds it along with the user query as input to the LLM. The LLM uses these cached context states to generate answers, completely bypassing the real-time retrieval step.  

- **Addressing the limitations of RAG** : CAG aims to address some of the core pain points of traditional RAG:
  - **Eliminate retrieval latency** : Since all knowledge is preloaded and cached, time-consuming real-time retrieval is not required during inference, significantly reducing response latency.
  - **Minimize retrieval errors**: Avoid retrieval errors (such as retrieving irrelevant or incomplete information) caused by imperfect retrieval algorithms, index construction problems, or query understanding deviations.
  - **Simplify system complexity**: Eliminate independent retrieval modules, vector databases, and the complex integration between the two and LLM, making the system architecture simpler and reducing development and maintenance costs.  

- **Dependencies** : The feasibility of CAG is highly dependent on whether the LLM has a long enough context window to accommodate the target knowledge base. With the emergence of models such as Llama 3.1 (128K context length), GPT-4 (128K), Claude 3 (200K) and even Gemini 1.5 Pro (1M tokens), it has become possible to process large amounts of text at once (such as entire document sets, internal company knowledge bases, FAQs, customer support logs, domain-specific databases, etc.).
The proposal of CAG actually reflects an important impact of the development of LLM's own capabilities on the RAG technology route. An important premise for the initial widespread research and application of RAG was that the context window of LLM at that time was relatively small and could not directly process large-scale external documents3 . Therefore, retrieval became a necessary means to filter out small segments of relevant text fragments from massive data for LLM to use. However, when the context window of LLM is expanded to the order of hundreds of thousands or even millions of tokens, for those application scenarios with relatively fixed and controllable knowledge scope (for example, questions and answers based on a specific manual, queries based on internal company rules and regulations, etc.), "instilling" all relevant information into LLM at one time and using its internal attention mechanism to locate and synthesize information may become a simpler and more direct solution than building a complex RAG pipeline.

This does not mean that RAG will be completely replaced. For scenarios where the scale of knowledge is extremely large (far exceeding the capacity of any LLM context window), the source of knowledge is highly dynamic, or complex multi-source information interaction is required, RAG and its various advanced optimization techniques still have irreplaceable value. However, the emergence of CAG reminds us that the technical route of RAG may be differentiated in the future: one route will continue to focus on the retrieval and generation optimization of ultra-large-scale and highly dynamic knowledge sources; the other route may focus on how to more efficiently utilize the long context capabilities of LLM to achieve a simplified version of RAG that "retrieves and caches once" or "injects full context", and in some cases does not even require the traditional "retrieval" step at all. This trend deserves close attention from researchers and practitioners in the field of RAG.

## 7. RAG system evaluation and challenges overcome
With the increasing complexity of RAG technology and the continuous expansion of its application areas, how to scientifically and comprehensively evaluate the performance of RAG systems and how to effectively overcome the challenges they face in actual deployment have become crucial issues.

### 7.1. Key indicators for evaluating RAG performance
The evaluation of a RAG system is a multi-dimensional problem, as its performance is affected by both the retrieval module and the generation module, and is closely related to specific application scenarios and user expectations. Evaluation usually requires examining the performance of the system's internal components as well as the overall end-to-end performance.

- **Retrieval component evaluation metrics**:
  - **Context Relevance/Precision** : Measures the degree to which the retrieved document blocks match the information needs of the user's query. That is, how much of the retrieved information is truly relevant.
  - **Context Recall** : Measures whether all necessary information needed to answer the question has been retrieved from the knowledge base.
  - **Comprehensiveness** : Evaluate whether the retrieved documents cover multiple aspects and different perspectives of the query topic.
  - **Correctness** : Evaluates the accuracy of the retrieved documents relative to a set of candidate documents, i.e., the system’s ability to identify and prioritize relevant documents.

- **Generate component evaluation metrics** :
  - **Faithfulness** : Measures the extent to which the generated answer accurately reflects the information in the retrieved document, that is, whether the answer is "faithful" to the context and does not fabricate or distort the facts.
  - **Answer Relevance** : Measures how closely the generated answer aligns with the intent and content of the user’s original query.
  - **Answer Correctness** : Measures the factual accuracy of the generated answer, usually by comparing it to a “gold standard” answer or fact.  

**Specific frameworks and indicators** :
  - **RAGAS (Retrieval Augmented Generation Assessment)** : This is a popular RAG evaluation framework that focuses on reference-free evaluation, that is, evaluating RAG quality without manually annotated gold standard answers. Its core metrics include fidelity, answer relevance, and context relevance.
  - **ASTRID (Automated and Scalable TRIaD for evaluating clinical QA systems)** : This is a RAG evaluation framework designed specifically for clinical QA scenarios. It contains three core indicators: context relevance (CR), refusal accuracy (RA, measuring the system's ability to correctly reject an answer when it cannot provide a safe or appropriate answer) and conversational faithfulness (CF, evaluating the accuracy of informative sentences in the answer and its consistency with the context, while considering the naturalness of the conversation).
  - **Bench-RAG** : This is a benchmarking pipeline for evaluating the effectiveness of Finetune-RAG, which uses GPT-4o as a referee LLM to evaluate the factual accuracy of the answers generated by the model when provided with both correct and fictitious contexts.

**Evaluation of upstream components** : The performance of the RAG system is also affected by upstream components such as chunking and embedding models. Therefore, the evaluation of these components is also important. For example, the evaluation of chunking methods can focus on keyword coverage, the minimum number of tokens required to answer questions, etc.; the evaluation of embedding models can refer to comprehensive benchmarks such as MTEB (Massive Text Embedding Benchmark) and MMTEB (Massive Multicultural Text Embedding Benchmark).

### 7.2. Ongoing Challenges and Limitations in Current RAG Implementations
Although RAG technology has made significant progress, it still faces many challenges and limitations in practical applications:

- **General Challenges** :
  - **Retrieval delays, errors and system complexity** : Real-time retrieval may introduce delays; retrieval errors (such as retrieving irrelevant or incomplete information) will directly affect the quality of generation; the integration and maintenance of the entire RAG system is also relatively complex.
  - **Scalability issues** : As the size of the knowledge base grows and the number of concurrent users increases, maintaining low latency and high throughput is a challenge.
  - **Bias propagation** : If the knowledge base or retrieval algorithm itself is biased, these biases may be amplified and reflected in the generated answers.
  - **Security and privacy** : When the RAG system is connected to a knowledge base containing sensitive information, it is necessary to ensure data security and user privacy is not leaked.
  - **Lack of explainability** : Understanding why RAG systems retrieve specific documents and how they generate specific responses based on those documents sometimes still lacks transparency.  


- **"Seven Points of Failure" proposed by Barnett et al .**: This study summarized seven common failure modes through case analysis of RAG systems in different fields:
  - **Missing Content** : The information required to answer the question does not exist in the knowledge base.
  - **Missed the Top Ranked Documents** : Relevant information exists but is not ranked high enough by the search algorithm.
  - **Not in Context** : Relevant information was retrieved but excluded from the final context passed to the LLM (eg due to length restrictions or consolidation strategies).
  - **Not Extracted** : The correct answer exists in the context provided to the LLM, but the LLM was not able to successfully extract it.
  - **Wrong Format** : LLM fails to generate answers in the specific format (e.g., table, list) requested by the user.
  - **Incorrect Specificity** : The answer is too general or too specific and does not meet the user's needs.
  - **Incomplete** : The answer is correct but omits relevant information.  

- **Specific technical challenges** :
  - **Efficiently handling multiple documents** : Even when controlling for the total context length, LLM still faces challenges in processing information scattered across multiple documents, which is a different problem from simply processing a single long document.
  - **Data consistency, model alignment, integration complexity, error handling** : these are common engineering challenges when building and maintaining a RAG Pipeline.  

RAG systems are becoming increasingly complex, and their application scenarios are becoming increasingly diverse and sophisticated (e.g., from simple question-answering to complex multi-hop reasoning, conversational interactions, etc.). This development trend places higher demands on evaluation methods and metrics. Early RAG evaluations may focus on basic retrieval precision and the factuality of answers. However, with the emergence of advanced RAG technologies, such as iterative retrieval, self-correction mechanisms, and the ability to handle specific scenarios (such as clinical question-answering), the dimensions of evaluation must also be expanded accordingly. For example, it is necessary to evaluate the quality of the system's multi-hop reasoning, the robustness of the answer generation when the retrieval results contain noise (as the critique mechanism of Self-RAG focuses on), or the ability to correctly reject answers when faced with inappropriate questions (such as the rejection accuracy metric in the ASTRID framework).

Frameworks like RAGAS [ragus github](https://github.com/explodinggradients/ragas) attempt to achieve reference-free evaluation, which is critical for real-world scenarios that lack gold standard answers. At the same time, specialized benchmarks for specific capabilities (such as long-context processing and clinical question answering) are also emerging. 

It can be said that there is a continuous "competition" relationship between the evolution of RAG technology and the development of evaluation methods. As the RAG technology route continues to innovate and deepen, the evaluation methodology must also develop accordingly to accurately and comprehensively measure the performance, reliability and potential risks of these new systems. Only by continuously improving the evaluation system can we promote the construction of truly powerful and trustworthy RAG applications.
