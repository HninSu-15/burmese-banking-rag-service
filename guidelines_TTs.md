Here is a brief and practical way to explain **how Person 1 and Person 2 will implement their parts** based on the proposal architecture. 

### Person 1 — RAG Retrieval + Qwen3 Embedding

**Main goal:** Find the most relevant information from approved banking documents.

**How to do it:**

* Collect approved files such as PDF, DOCX, PPTX, images, and FAQs.
* Extract text from documents.
* Use **PaddleOCR** for scanned PDFs or images.
* Clean and normalize Burmese text.
* Split large documents into smaller **chunks**.
* Add metadata such as document name, page, section, and version.
* Use **Qwen3 Embedding** to convert each chunk into vectors.
* Store vectors and metadata in **ChromaDB**.
* When a user asks a question:

  * Convert the question into an embedding using Qwen3.
  * Search ChromaDB using vector similarity.
  * Return the top relevant chunks to Person 2. 

**Simple flow:**

```text
Documents
   ↓
Text Extraction / OCR
   ↓
Cleaning
   ↓
Chunking
   ↓
Qwen3 Embedding
   ↓
ChromaDB

User Question
   ↓
Query Embedding
   ↓
Vector Search
   ↓
Relevant Documents
```

**Main technologies:**

* Qwen3 Embedding
* ChromaDB
* LlamaIndex
* PaddleOCR
* PyMuPDF / pdfplumber

**Final output:**

```text
Relevant document chunks
+
Source information
+
Similarity score
```

Example:

```json
{
  "text": "If an ATM card is lost, contact the bank immediately...",
  "document": "ATM_Card_Policy.pdf",
  "page": 5,
  "score": 0.92
}
```

---

### Person 2 — Conversation Manager + Gemini LLM

**Main goal:** Use the retrieved information to generate a correct Burmese answer.

**How to do it:**

* Receive the user's question.
* Create or maintain the **conversation session**.
* Store short-term conversation history.
* Use previous messages to understand follow-up questions.
* Send the user question to Person 1's retrieval component.
* Receive relevant banking document chunks.
* Build a prompt containing:

  * user question
  * retrieved documents
  * conversation history
  * banking rules
  * Burmese response instructions
* Send that prompt to **Gemini LLM**.
* Generate a grounded Burmese answer.
* If the retrieved documents do not contain enough information, the system should avoid guessing and return an unsupported-answer response.
* Normalize the Burmese output for consistency. 

**Simple flow:**

```text
User Question
     ↓
Session
     ↓
Conversation History
     ↓
Send Query to Person 1
     ↓
Retrieved Documents
     ↓
Prompt Builder
     ↓
Gemini LLM
     ↓
Grounded Burmese Answer
     ↓
Burmese Text Normalization
```

**Prompt example:**

```text
System:
You are a Burmese banking support assistant.
Only answer using the provided banking documents.

Conversation History:
User previously asked about ATM card replacement.

Retrieved Context:
[Relevant ATM policy]

User Question:
ဘယ်လောက်ကုန်ကျမလဲ။

Instruction:
Answer clearly in Burmese.
```

**Main technologies:**

* Gemini LLM
* FastAPI
* Conversation/session handling
* Prompt Builder
* Burmese text normalization

**Final output:**

```text
Normalized Grounded Burmese Answer
```

---

### Very short team summary

| Person       | Main Job                  | Main Model          | Input                              | Output                   |
| ------------ | ------------------------- | ------------------- | ---------------------------------- | ------------------------ |
| **Person 1** | Search banking knowledge  | **Qwen3 Embedding** | User query + documents             | Relevant document chunks |
| **Person 2** | Generate Burmese response | **Gemini LLM**      | Query + retrieved chunks + history | Grounded Burmese text    |

The connection between them is simply:

```text
Person 2
User Question
     ↓
Person 1
Qwen3 + ChromaDB
     ↓
Relevant Documents
     ↓
Person 2
Gemini
     ↓
Grounded Burmese Answer
```

For implementation, **Person 1 should first make `retrieve(query)` work independently**, while **Person 2 should make `generate_answer(query, retrieved_chunks, history)` work independently**. After both work separately, connect the two components.
