# Swedish Medication Interaction Checker


This README explains the work completed for the Data and RAG part of the Swedish Medication Interaction Checker project.

The main goal of my part was to collect, clean, structure, store, retrieve, and evaluate medication interaction data from FASS.

1. Data Collection
I started with medication data collected from FASS.

Main file used:
fass_444_normalized.csv

This file contained:
- ATC code
- active ingredient
- product name
- interaction text
- FASS source URL

2. Data Cleaning
I cleaned the interaction data by:
- removing empty interaction rows
- keeping useful medication information
- cleaning extra spaces
- preparing the text for further processing

I also selected one representative interaction text for each ATC code.

3. Extracting FASS Section 4.5
For this project, I extracted:
4.5 Interaktioner med andra läkemedel och övriga interaktioner

This is the section that contains medication interaction information.

I also checked that other sections such as:
4.4 Varningar
4.6 Graviditet
4.8 Biverkningar
5.1 Farmakodynamik

were not incorrectly included.

4. Splitting Interaction Text into Statements
The section 4.5 text was often very long, so I split it into smaller sentences or statements.

Example:
Erytromycin hämmar metabolismen av sildenafil.
En startdos på 25 mg sildenafil bör övervägas.

This made it easier to identify individual interaction information.

5. Detecting Drug Mentions
I created a drug lookup using:
- drug names
- ATC codes
- some drug class aliases

Examples:
NSAID
ACE-hämmare
statiner
orala antikoagulantia
antidiabetika

The system searched each interaction statement for possible Drug B mentions.

6. Detecting Clear Interaction Relationships
Not every drug name mentioned in a FASS statement means there is a real interaction.

To reduce false positives, I used relation patterns such as:
samtidig användning
administreras samtidigt
bör undvikas
rekommenderas inte
kontraindicerad
dosjustering
bör övervakas

A pair was only considered high-confidence when Drug A, Drug B, and a clear interaction relation were present.

7. Hybrid Interaction Strategy
I used a hybrid strategy.

If the interaction between two drugs was clear:
Drug A + Drug B -> structured interaction pair

If a statement did not contain a clear enough relation:
original FASS statement -> saved as fallback RAG evidence

This helped reduce false drug pairs while still keeping important FASS information.

8. Adjacent Statement Context
Sometimes FASS explains one interaction across multiple sentences.

Example:
Erytromycin hämmar sildenafils metabolism.
En startdos på 25 mg sildenafil bör övervägas.

I added adjacent-statement context merging so these two sentences could stay together.

9. Severity Assignment
I created simple rule-based severity labels:
DANGEROUS
AVOID
CAUTION
INFO

Examples:
kontraindicerad -> DANGEROUS
bör undvikas / rekommenderas inte -> AVOID
försiktighet / dosjustering / bör övervakas -> CAUTION
no strong warning keyword -> INFO

Important:
INFO does NOT mean SAFE.
It only means that no stronger severity keyword was detected.

10. Creating Final Interaction Chunks
I created two main types of chunks.

Structured pair chunks:
fass_pair_chunks.csv

Fallback interaction chunks:
fass_fallback_chunks.csv

11. Creating Embeddings
I used:
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

The model converted the interaction chunks into vector embeddings.

Embedding size:
384 dimensions

12. Building ChromaDB
I created a persistent ChromaDB database.

Database folder:
drug_db/

Collection name:
medication_interactions

The database contains:
interaction_pair
interaction_statement

Build script:
build_chromadb.py

13. Retrieval Strategy
The final retrieval system works like this:

Drug A + Drug B
-> ATC codes
-> Exact structured pair lookup

If pair found:
return structured FASS evidence

If pair not found:
filter fallback statements by Drug A ATC
-> semantic search
-> Drug B reranking
-> return best FASS statements

14. Exact ATC Pair Retrieval
If both drug ATC codes are known, the system first checks for an exact structured interaction pair.

Example:
Rimegepant (N02CD06)
+
Modafinil (N06BA07)

The system directly found:
Rimegepant + modafinil
Severity: AVOID

15. Fallback Semantic Retrieval
Sometimes a structured pair does not exist.

In that case, the system searches the original FASS interaction statements.

The fallback search:
1. filters by Drug A ATC
2. performs semantic search
3. retrieves more candidates
4. gives priority to results that explicitly mention Drug B

Example:
Digoxin + metoklopramid

The final retrieval returned at Rank 1:
Motilitetsstimulerande läkemedel, så som metoklopramid, kan minska absorptionen av digoxin.

16. Retrieval Evaluation
I tested the retrieval system using known interaction cases.

Structured retrieval evaluation:
20 known structured drug pairs
Result:
20/20 successfully retrieved
All exact structured pairs were found at Rank 1.

This means 100% success on this 20-case evaluation set.
It does NOT mean 100% accuracy for the full system.

17. Fallback Retrieval Evaluation
I also tested 20 fallback-only interaction cases.

Before improvements:
Hit@5 = 45%
Hit@10 = 60%
MRR = 0.314

After filtering by Drug A ATC:
Hit@5 = 65%
Hit@10 = 80%
MRR = 0.597

After adding Drug B reranking:
Hit@5 = 20/20 = 100%
Hit@10 = 20/20 = 100%
MRR = 0.938

This shows that ATC filtering and reranking improved retrieval performance on the test set.

18. Meaning of Evaluation Metrics
Hit@5:
Checks whether the correct interaction appears within the top 5 retrieved results.

Hit@10:
Checks whether the correct interaction appears within the top 10 retrieved results.

MRR:
Mean Reciprocal Rank.
It gives a higher score when the correct result appears closer to Rank 1.

19. Main Python Files
build_chromadb.py
- builds the ChromaDB database

retrieval.py
- exact ATC lookup
- fallback semantic retrieval
- ATC filtering
- Drug B reranking
- hybrid retrieval

evaluate_20_pairs.py
- structured interaction evaluation

evaluate_fallback_20.py
- fallback retrieval evaluation

debug_fallback_failures.py
- used to inspect failed retrieval cases

20. Main Data Files
fass_pair_chunks.csv
fass_fallback_chunks.csv
retrieval_test_20_pairs.csv
retrieval_evaluation_results_exact_hybrid.csv
fallback_test_20.csv
fallback_evaluation_results_final.csv

21. How you Can Use My Work
Mainly needs:
retrieval.py
drug_db/
requirements.txt

Example:

from retrieval import hybrid_search

results = hybrid_search(
    drug_a="Rimegepant",
    atc_a="N02CD06",
    drug_b="modafinil",
    atc_b="N06BA07"
)

The retrieved FASS evidence can then be sent to the LLM.

You can use this evidence for:
- interaction explanation
- severity presentation
- citations
- safety message
- final chatbot response

22. Requirements
Install dependencies using:
python -m pip install -r requirements.txt

Main packages:
pandas
numpy
chromadb
sentence-transformers
mistralai
python-dotenv
streamlit

23. Rebuilding the Database
If drug_db/ is missing:
python build_chromadb.py

This will:
load pair chunks
+
load fallback chunks
-> create embeddings
-> store them in ChromaDB

24. Final Status
My work includes:
- Data collection
- Data cleaning
- FASS section 4.5 extraction
- Statement splitting
- Drug mention detection
- Interaction relation detection
- High-confidence drug pair creation
- Fallback statement preservation
- Adjacent context merging
- Severity assignment
- Chunk generation
- Embedding generation
- ChromaDB creation
- Exact ATC retrieval
- Semantic fallback retrieval
- ATC filtering
- Drug-B reranking
- Retrieval testing
- Retrieval evaluation

The Data and RAG layer is now ready for integration with the LLM part of the project.
