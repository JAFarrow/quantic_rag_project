# AI Usage

AI tools were utilized across the stack, mostly in the form of ChatGPT and OpenCode utilizing GPT-5.3 Codex as its' model.

## Planning and Architectural/Tech Stack Decisions

Reasoned over the project spec alongside ChatGPT before technical implementation as to potential risks and tradeoffs with the initial tech stack I had in mind. 

Utilized ChatGPT to summarize and advise on Pinecone integration. 

- **What went well**: Fast iteration on design tradeoffs and clearer up-front framing of Pinecone fit, cost, and complexity before writing code.
- **What didn't**: Advice quality was highly prompt-dependent and occasionally generic, so recommendations still needed manual validation against project constraints.

## FastAPI Backend

Initial repo setup and bootstrapping via OpenCode agentic development. 

- **What went well**: Bootstrapping and scaffolding velocity was high, especially for initial routing, structure, and repetitive setup tasks.
- **What didn't**: -

## Static Frontend

Entirely generated with context of the backend implementation via OpenCode agentic development.

- **What went well**: Rapid UI generation with backend-aware context reduced integration friction and enabled quick iteration on end-to-end flows.
- **What didn't**: Generated output occasionally favored generic patterns and needed manual refinement for visual polish, copy tone, and responsive behavior details.

## Testing and Evaluation

Instructed GPT5.3-Codex model via OpenCode to consume the policy documents in data, take a list of questions and test against a local instance of the backend and provide the evaluation output as found in design-and-evaluation.md

- **What went well**: Automated evaluation loop made it straightforward to run broad question sets and produce a reproducible baseline quickly.
- **What didn't**: Result quality remained sensitive to question phrasing and retrieval variance, requiring manual spot checks to distinguish model misses from data/retrieval issues.
