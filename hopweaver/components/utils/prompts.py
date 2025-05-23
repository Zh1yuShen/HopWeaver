"""
This file contains all prompts used in the HopWeaver system.
Centralizing prompts makes it easier to manage and adjust them.
"""

# Evaluation prompt template
# Bridge question evaluation - Question only mode
BRIDGE_QA_ONLY_PROMPT = """You are an AI assistant designed for exact answer extraction.

IMPORTANT INSTRUCTIONS FOR ANSWERING:
1. Provide ONLY the exact answer - a name, date, term, or short phrase.
2. DO NOT write complete sentences or explanations.
3. DO NOT use punctuation at the end of your answer.
4. DO NOT include articles like 'the', 'a', or 'an' unless they are part of a name.
5. DO NOT use formatting like bold, italics, or bullet points.
6. If the answer is a date, use the format: Month Day, Year (e.g. April 16, 1934).
7. If the answer is a person's name, provide only their full name.
8. If the answer is a place, provide only the place name.
9. Answer in English ONLY.
10. Keep your answer to 10 words or fewer whenever possible.

Examples:
- Question: When was Albert Einstein born?
  Correct: March 14, 1879
  Incorrect: Albert Einstein was born on March 14, 1879.

- Question: Who discovered penicillin?
  Correct: Alexander Fleming
  Incorrect: The person who discovered penicillin was Alexander Fleming.

Your goal is to match the exact expected answer format."""

# Bridge question evaluation - Question + Document mode
BRIDGE_QA_DOCS_PROMPT = """You are an AI assistant designed for exact answer extraction from documents.

IMPORTANT INSTRUCTIONS FOR ANSWERING:
1. Provide ONLY the exact answer - a name, date, term, or short phrase.
2. DO NOT write complete sentences or explanations.
3. DO NOT use punctuation at the end of your answer.
4. DO NOT include articles like 'the', 'a', or 'an' unless they are part of a name.
5. DO NOT use formatting like bold, italics, or bullet points.
6. If the answer is a date, use the format: Month Day, Year (e.g. April 16, 1934).
7. If the answer is a person's name, provide only their full name.
8. If the answer is a place, provide only the place name.
9. Answer in English ONLY.
10. Keep your answer to 10 words or fewer whenever possible.

Examples:
- Question: When was the Declaration of Independence signed?
  Correct: August 2, 1776
  Incorrect: The Declaration of Independence was signed on August 2, 1776.

- Question: Who was the first president of the United States?
  Correct: George Washington
  Incorrect: Based on the documents, George Washington was the first president.

Your goal is to match the exact expected answer format."""

# Comparison question evaluation - Question only mode
COMPARISON_QA_ONLY_PROMPT = """You are an AI assistant designed for exact answer extraction.

IMPORTANT INSTRUCTIONS FOR ANSWERING:
1. Provide ONLY the exact answer - a name, date, term, or short phrase.
2. DO NOT write complete sentences or explanations.
3. DO NOT use punctuation at the end of your answer.
4. DO NOT include articles like 'the', 'a', or 'an' unless they are part of a name.
5. DO NOT use formatting like bold, italics, or bullet points.
6. If the answer is a date, use the format: Month Day, Year (e.g. April 16, 1934).
7. If the answer is a person's name, provide only their full name.
8. If the answer is a place, provide only the place name.
9. Answer in English ONLY.
10. Keep your answer to 10 words or fewer whenever possible.

Examples:
- Question: When was Albert Einstein born?
  Correct: March 14, 1879
  Incorrect: Albert Einstein was born on March 14, 1879.

- Question: Who discovered penicillin?
  Correct: Alexander Fleming
  Incorrect: The person who discovered penicillin was Alexander Fleming.

Your goal is to match the exact expected answer format."""

# Comparison question evaluation - Question + Document mode
COMPARISON_QA_DOCS_PROMPT = """You are an AI assistant designed for exact answer extraction from documents.

IMPORTANT INSTRUCTIONS FOR ANSWERING:
1. USE ONLY INFORMATION FROM THE PROVIDED DOCUMENTS to determine your answer.
2. Provide ONLY the exact answer - a name, date, term, or short phrase.
3. DO NOT write complete sentences or explanations.
4. DO NOT use punctuation at the end of your answer.
5. DO NOT include articles like 'the', 'a', or 'an' unless they are part of a name.
6. DO NOT use formatting like bold, italics, or bullet points.
7. If the answer is a date, use the format: Month Day, Year (e.g. April 16, 1934).
8. If the answer is a person's name, provide only their full name.
9. If the answer is a place, provide only the place name.
10. Answer in English ONLY.
11. Keep your answer to 10 words or fewer whenever possible.

Examples:
- Question: When was the Declaration of Independence signed?
  In document: "The Declaration of Independence was signed on August 2, 1776 in Philadelphia."
  Correct: August 2, 1776
  Incorrect: The Declaration of Independence was signed on August 2, 1776.

- Question: Who was the first president of the United States?
  In document: "George Washington served as the first president from 1789 to 1797."
  Correct: George Washington
  Incorrect: Based on the documents, George Washington was the first president.

Your goal is to match the exact expected answer format while ensuring accuracy based on the documents."""

# Entity extraction prompts
ENTITY_EXTRACTION_PROMPT = """###---Goal---

Given a text document, select a single segment (e.g., a paragraph or sentence) with high potential to contain a bridge entity for multi-hop question generation, identify one bridge entity from that segment, extract relevant text segments, and generate an expanded query statement for this bridge entity to retrieve related documents from a vector database.

### ---Instructions---

1. **Select a Segment and Identify a Bridge Entity**  
   Select a single text segment from the document that appears to have high potential for containing a bridge entity. Then, identify one bridge entity from this segment. A bridge entity can be any noun (e.g., a person, place, time, event, object, concept, or other noun categories), and should satisfy the following principles:  
   - **High Connectivity**: The entity has multiple associations with other entities (e.g., people, events, dates, locations, ideas), making it likely to retrieve related documents.  
   - **Uniqueness and Clarity**: The entity is clearly defined within the segment, avoiding ambiguity or multiple interpretations.  
   - **Attribute Richness**: The entity has multiple queryable attributes (e.g., creator, date, location, impact, participants), supporting diverse multi-hop questions.  
   - **Cross-Document Distribution Potential**: The entity's information is likely partially described in the segment, with additional details potentially spread across other documents.  
   - **Distinct from Title**: The bridge entity must not be identical to the document title ({document_title}), but it should be thematically related to the text's main topic to enable multi-hop connections to other relationships.  
   - **Information-Rich Segment**: Prefer segments that are rich in information about the bridge entity, avoiding overly fragmented or sparse divisions.  

   From the chosen segment, extract:  
   - **entity_name**: Name of the entity, capitalized.  
   - **entity_type**: A general type reflecting its noun category (e.g., Person, Location, Time, Event, Object, Concept). Provide your best guess if unsure.  

   Format as: ("bridge_entity"<|>"entity_name"<|>"entity_type")

2. **Extract Relevant Text Segments**  
   For the bridge entity identified in step 1, extract a single part of the document that directly mentions or describes the entity, focusing on the information from the selected segment and maximizing the inclusion of related details. 
   
   Before presenting the extracted segment, provide a single brief sentence that identifies the main entity and its basic role/context - just enough to prevent ambiguity when reading the segment.
   
   The extracted part should:
   - Be the most information-rich portion available, comprehensively reflecting the bridge entity's characteristics, context, and associations (e.g., causes, consequences, or attributes) as presented in the document.  
   - Only one part is allowed, so ensure it contains sufficient information to stand alone as a complete representation of the entity within the text; this part may span multiple sentences or a paragraph, depending on content coherence.  
   - While ensuring complete introduction of the entity and its relevant information, avoid being overly lengthy, preferably containing between 50-200 words.
   - Carefully check the part for all pronouns (e.g., "he", "his", "she", "it", "they", "their", etc.). If any pronoun's reference is unclear or ambiguous (i.e., not immediately obvious within the part itself), replace or rephrase it with an explicit reference (e.g., the entity name or a clear descriptor) to ensure the part is independently readable and contains complete, self-contained information.  
   - **entity_name**: The name of the bridge entity.  
   - **relevant_segments**: The contextual introduction followed by the extracted segment, with all unclear pronouns resolved, enclosed in double quotes.

   Format as: ("relevant_segments"<|>"entity_name"<|>"entity_introduction + extracted_part")

3. **Generate an Expanded Query Statement**  
   For the bridge entity identified in step 1, generate one expanded query statement to retrieve related documents from a vector database for multi-hop question generation. The query should:  
   - Include the **entity_name** and **entity_type**.
   - Focus on retrieving information that is COMPLEMENTARY to what is already in the document, not redundant information.
   - Use semantic direction shifting phrases like "instead of," "beyond," "outside of," or "rather than" to steer the vector search away from the original document's content.
   - Incorporate keywords about aspects, roles, or contexts of the entity NOT covered in the original document.
   - Use domain transfer wording to explore the entity in completely different contexts than those mentioned in the document.
   - Remain concise (preferably 10-20 words) while being semantically rich and directionally distinct.
   - Example format: "[entity_name] [alternate aspect] instead of [aspect covered in document]"
   Format as: ("query"<|>"entity_name"<|>"entity_query")

4. **Return Output**  
   Return output in English as a single list containing the identified bridge entity, its relevant segments, and its expanded query, following the formats in steps 1, 2, and 3. Use " ## " as the list delimiter between entity, segments, and query.  

5. **When Finished**  
   Output <|COMPLETE|>

---Real Data---
##################
Title: {document_title}
Text: {input_text}
##################
The "Output" should follow the exact format specified in the instructions (DON'T use JSON format).

Output:
"""

# Comparison type entity extraction prompt
COMPARE_ENTITY_EXTRACTION_PROMPT = """###---Goal---

Given a text document, identify its primary subject entity and extract multiple key attributes associated with this entity, along with their corresponding values. For each extracted attribute, generate an expanded query statement designed to retrieve documents about *other* similar entities that also possess this attribute, facilitating subsequent comparison.

### ---Instructions---

1.  **Identify the Primary Subject Entity**
    *   Determine the main person, place, organization, event, concept, or work (e.g., book, film) that the document ({document_title}) is primarily about. This is often the title itself or the entity most central to the text.
    *   Determine the **subject_entity_name** (capitalized) and its general **subject_entity_type** (e.g., Person, Location, Organization, Event, Film, Concept).

2.  **Extract Comparable Attributes, Values, and Generate Queries**
    *   Identify **multiple (aim for 3-5 if possible)** distinct attributes or properties associated with the primary subject entity identified in step 1, directly stated in the document text ({input_text}).
    *   **Focus strictly on attributes whose VALUES are suitable for comparison.** This means the value should be relatively concise, factual, and belong to a common data type. Prioritize attributes that meet these criteria:
        *   **Concise & Factual Value:** Short value (e.g., name, number, date, category, location).
        *   **Common Data Types:** Prefer Numbers, Dates, Locations, Specific Names (Entities), Defined Categories/Types.
        *   **Likely Commonality:** Prefer attributes likely to exist for other entities of the same `subject_entity_type`.
    *   **Source Preference:** Give priority to attributes explicitly mentioned in introductory paragraphs or summary sections.
    *   **Avoid:** Attributes requiring subjective judgment, having vague/ambiguous values, or whose values are long narrative descriptions.
    *   **For each identified comparable attribute:**
        *   a. Determine the **attribute_name** (e.g., "Population", "Date of Birth", "Director").
        *   b. Extract the **attribute_value** (e.g., "1.2 million", "1990-05-15", "Christopher Nolan").
        *   c. Generate an **entity_b_query**: A concise (10-20 words) query suitable for vector database retrieval, designed to find *other* entities of the same `subject_entity_type` that also possess the `attribute_name`. Use phrasing to encourage diversity (e.g., "Other [type] with known [attribute]", "Examples of [type] and their [attribute]").

3.  **Final Check & Completion Signal**
    *   Review the generated subject entity and attribute tuples for accuracy and adherence to instructions.
    *   Signal completion once all steps are done.

### ---Output Format Specification---

**Strictly adhere to the following output format:**

1.  **Structure:** The entire output must be a single string. It starts with the subject entity part, optionally followed by one or more attribute parts.
2.  **Subject Entity Part:** The first part MUST be formatted as: `("subject_entity"<|>"subject_entity_name"<|>"subject_entity_type")`.
3.  **Attribute Parts:** If comparable attributes are found, each subsequent part MUST be formatted as: `("attribute"<|>"attribute_name"<|>"attribute_value"<|>"entity_b_query")`.
4.  **Delimiter:** Use ` ## ` (space, hash, hash, space) strictly as the delimiter *between* parts (i.e., between the subject part and the first attribute part, and between consecutive attribute parts). **Do not** use it at the very beginning or very end.
5.  **Edge Cases:**
    *   If no comparable attributes are found, output *only* the subject entity part.
    *   If the subject entity is ambiguous, output *only* the string `("subject_entity"<|>"Subject Ambiguous"<|>"Unknown")`.
6.  **Completion Signal:** Append `<|COMPLETE|>` to the very end of the entire generated string.

---Real Data---
##################
Title: {document_title}
Text: {input_text}
##################
The "Output" should follow the exact format specified in the `---Output Format Specification---` section.

Output:
"""

# Sub-question generation prompt
SUB_QUESTION_GENERATION_PROMPT = """---Goal---
Analyze two documents connected by a bridge entity and generate two sequential sub-questions that form a multi-hop reasoning chain.

---Instructions---
Analyze how the bridge entity connects both documents by:
- Identifying key information about the bridge entity in Document A that is unique to Document A (not mentioned or implied in Document B).
- Finding related information in Document B that connects via this bridge entity and is unique to Document B (not mentioned or implied in Document A).
- Determining a clear reasoning path where the unique information from Document A leads to the unique information in Document B.
- If Documents A and B do not form a valid bridge entity connection, or if Document B's content appears to describe a different entity with a similar name but significantly different characteristics (entity disambiguation issue), return an error identifier "INVALID_BRIDGE_CONNECTION" along with a brief explanation of why the connection is invalid.

Generate two sequential sub-questions:
- Sub-question 1: A question about Document A where the answer is the bridge entity, using only information exclusive to Document A. The bridge entity should not appear in the question's wording.
- Sub-question 2: A question that explicitly uses the bridge entity (from the answer to Sub-question 1) in its wording to find related information in Document B, using only information exclusive to Document B.

Each sub-question must:
- Be answerable from only one document.
- Have a definitive answer contained in its document, based on information that does not appear or cannot be inferred from the other document.
- Be phrased as a standalone question without phrases like "According to Document A/B" or similar document references.
- Be specific with clear and unambiguous references to information in its respective document.
- Provide an answer that is clear, concise, and not a full sentence (e.g., a name, number, or short phrase).
- Together form a logical reasoning chain where the answer to Sub-question 1 (the bridge entity or its unique information) is necessary to answer Sub-question 2.

Format your output exactly as follows:

<!-- If no valid bridge connection exists -->
INVALID_BRIDGE_CONNECTION
Reason: [Brief explanation why the documents cannot form a valid bridge connection]

<!-- If valid bridge connection exists -->
ANALYSIS:
Bridge connection: [How the bridge entity connects the documents]
Document A segments: [Copy of the original Document A segments provided in the input]
Document B segments: [Relevant excerpts from Document B that connect to the bridge entity, structured similarly to Document A segments, preferably containing between 50-200 words to ensure complete information while avoiding excessive length]
Reasoning path: [Logical path from Document A to Document B]

SUB-QUESTIONS:
Sub-question 1: [Question about Document A]
Answer 1: [Answer from Document A - about the bridge entity]

Sub-question 2: [Question using bridge entity to find answer in Document B]
Answer 2: [Answer from Document B]

---Real Data---
##################
Bridge Entity: {bridge_entity}
Entity Type: {entity_type}

Document A (Relevant Segments):
{doc_a_segments}

Document B (Retrieved Document):
{doc_b_document}
##################

Output:
"""

# Comparison type entity extraction prompt
COMPARE_ENTITY_EXTRACTION_PROMPT = """###---Goal---

Given a text document, identify its primary subject entity and extract multiple key attributes associated with this entity, along with their corresponding values. For each extracted attribute, generate an expanded query statement designed to retrieve documents about *other* similar entities that also possess this attribute, facilitating subsequent comparison.

### ---Instructions---

1.  **Identify the Primary Subject Entity**
    *   Determine the main person, place, organization, event, concept, or work (e.g., book, film) that the document ({document_title}) is primarily about. This is often the title itself or the entity most central to the text.
    *   Determine the **subject_entity_name** (capitalized) and its general **subject_entity_type** (e.g., Person, Location, Organization, Event, Film, Concept).

2.  **Extract Comparable Attributes, Values, and Generate Queries**
    *   Identify **multiple (aim for 3-5 if possible)** distinct attributes or properties associated with the primary subject entity identified in step 1, directly stated in the document text ({input_text}).
    *   **Focus strictly on attributes whose VALUES are suitable for comparison.** This means the value should be relatively concise, factual, and belong to a common data type. Prioritize attributes that meet these criteria:
        *   **Concise & Factual Value:** Short value (e.g., name, number, date, category, location).
        *   **Common Data Types:** Prefer Numbers, Dates, Locations, Specific Names (Entities), Defined Categories/Types.
        *   **Likely Commonality:** Prefer attributes likely to exist for other entities of the same `subject_entity_type`.
    *   **Source Preference:** Give priority to attributes explicitly mentioned in introductory paragraphs or summary sections.
    *   **Avoid:** Attributes requiring subjective judgment, having vague/ambiguous values, or whose values are long narrative descriptions.
    *   **For each identified comparable attribute:**
        *   a. Determine the **attribute_name** (e.g., "Population", "Date of Birth", "Director").
        *   b. Extract the **attribute_value** (e.g., "1.2 million", "1990-05-15", "Christopher Nolan").
        *   c. Generate an **entity_b_query**: A concise (10-20 words) query suitable for vector database retrieval, designed to find *other* entities of the same `subject_entity_type` that also possess the `attribute_name`. Use phrasing to encourage diversity (e.g., "Other [type] with known [attribute]", "Examples of [type] and their [attribute]").

3.  **Final Check & Completion Signal**
    *   Review the generated subject entity and attribute tuples for accuracy and adherence to instructions.
    *   Signal completion once all steps are done.

### ---Output Format Specification---

**Strictly adhere to the following output format:**

1.  **Structure:** The entire output must be a single string. It starts with the subject entity part, optionally followed by one or more attribute parts.
2.  **Subject Entity Part:** The first part MUST be formatted as: `("subject_entity"<|>"subject_entity_name"<|>"subject_entity_type")`.
3.  **Attribute Parts:** If comparable attributes are found, each subsequent part MUST be formatted as: `("attribute"<|>"attribute_name"<|>"attribute_value"<|>"entity_b_query")`.
4.  **Delimiter:** Use ` ## ` (space, hash, hash, space) strictly as the delimiter *between* parts (i.e., between the subject part and the first attribute part, and between consecutive attribute parts). **Do not** use it at the very beginning or very end.
5.  **Edge Cases:**
    *   If no comparable attributes are found, output *only* the subject entity part.
    *   If the subject entity is ambiguous, output *only* the string `("subject_entity"<|>"Subject Ambiguous"<|>"Unknown")`.
6.  **Completion Signal:** Append `<|COMPLETE|>` to the very end of the entire generated string.

---Real Data---
##################
Title: {document_title}
Text: {input_text}
##################
The "Output" should follow the exact format specified in the `---Output Format Specification---` section.

Output:
"""

# Multi-hop question synthesis prompt
MULTI_HOP_QUESTION_SYNTHESIS_PROMPT = """---Goal---
Synthesize a concise, natural multi-hop question that requires reasoning across two documents, connecting two sub-questions into a single logical inquiry.

---Instructions---
- FIRST, check if Answer 1 (from the first sub-question) is included in the text of the second sub-question. If it is NOT included, return "NONE" as the result and explain that the bridge entity wasn't properly utilized in the second question.
- Review the analysis and sub-questions to trace the full reasoning chain, identifying the bridge entity and unique information from each document.
- Create a single multi-hop question that:
  - Is ONE cohesive question, not multiple questions combined or concatenated
  - Requires distinct information from both Document A and Document B to answer
  - Reads naturally as a coherent, conversational question
  - Cannot be fully answered using only one document
  - Follows the reasoning path of the sub-questions, using the bridge entity from Sub-question 1's answer to link to Sub-question 2's information
  - Is clear, concise, and free of ambiguity
  - Doesn't explicitly mention the bridge entity or intermediate reasoning steps (these should be discovered through reasoning)
- If the two sub-questions cannot be combined into a valid multi-hop question that meets all the above criteria, return "NONE" as the result with a brief explanation of why.
- Ensure:
  - The answer matches Answer 2 from the sub-questions
  - The reasoning path integrates the unique contributions of both documents, as outlined in the analysis

Format your output exactly as follows:

<!-- If sub-questions cannot be combined into a valid multi-hop question -->
NONE
Reason: [Brief explanation why a valid multi-hop question cannot be created]

<!-- If a valid multi-hop question can be created -->
MULTI-HOP QUESTION: [Your synthesized question]

ANSWER:
[The final answer, matching Answer 2]

REASONING PATH:
[Step-by-step explanation of the multi-hop process, showing how each document contributes]

SOURCES:
[Document A and Document B, specifying their roles]

---Real Data---
##################
ANALYSIS:
{analysis}

SUB-QUESTIONS:
{sub_questions}
##################

Output:
"""

# Polisher module prompt for multi-hop question validation and refinement
POLISHER_PROMPT = """---Goal---
Validate and refine multi-hop questions to ensure they genuinely require cross-document reasoning and follow a proper reasoning chain where information from one document is essential to answer a question about content in another document.

---Instructions---
You are a Polisher module responsible for validating and refining multi-hop questions. Given a multi-hop question, its suggested answer, reasoning path, and source document segments, you will evaluate the question's quality and make one of four decisions:

1. **PASS**: The question is valid, well-formed, and genuinely requires both documents.
2. **ADJUST**: The question needs surface wording improvements only.
3. **REWORKED**: The question needs substantial structural changes.
4. **REJECTED**: The question has unfixable flaws.

Review and modify the question based on these key dimensions:

1. **True Multi-hop Necessity**: CRITICAL - A proper multi-hop question requires:
   - Information must flow from Document A to Document B in a logical sequence
   - The answer must be impossible to determine using either document in isolation
   - The reasoning path must demonstrate how Document A provides context necessary for Document B
   - The question should require discovering connections that aren't explicitly stated in either document

2. **Hidden Bridge Structure**:
   - The question should NOT directly mention the connecting entity or concept
   - The bridge entity should remain implicit in the question wording
   - The question should require identifying the relevant bridge entity as part of the reasoning process
   - Reframe questions that explicitly name the bridge entity to make the reasoning more challenging

3. **Reasoning and Answer Quality**:
   - Verify the reasoning follows a logical progression from Document A to Document B
   - Ensure the answer is factually accurate according to both documents
   - Check that the answer requires synthesizing information across documents
   - Improve question wording for clarity, fluency, and natural conversational tone
   - Remove any hints in the question that reveal the reasoning steps

---Output Formats---

IMPORTANT: Only output the exact content requested below WITHOUT any explanations, justifications, or additional text. Do not include your thought process, analysis, or reasoning. Only produce the specified outputs in the exact format shown.

### 1. If the question passes all criteria without changes:
[PASS]

### 2. If the question needs minor adjustments:
[ADJUST]

REFINED_REASONING_PATH: [Updated reasoning path that connects the two documents]

REFINED_QUESTION: [Adjusted question with minor improvements]

REFINED_ANSWER: [Updated answer if needed, otherwise keep the original]


### 3. If the question needs significant refinement:
[REWORKED]

REFINED_REASONING_PATH: [Completely revised reasoning path]

REFINED_QUESTION: [Substantially revised question]

REFINED_ANSWER: [Updated answer based on the revised question]


### 4. If the question is fundamentally flawed:
[REJECTED]


---Real Data---
##################
Multi-hop Question: {multi_hop_question}
Answer: {answer}
Reasoning Path: {reasoning_path}

Sub-question 1 (for Document A) with answer: {sub_question_1}
Sub-question 2 (for Document B) with answer: {sub_question_2}

(These sub-questions and their answers show the key information that should be extracted from each document)

Document A Segment:
{doc_a_seg}

Document B Segment:
{doc_b_seg}
##################

Output:
"""

# MHQA_QUALITY_ASSESSMENT_PROMPT
MHQA_QUALITY_ASSESSMENT_PROMPT = """---Goal---
You are tasked with conducting a **rigorous and critical** evaluation of multi-hop questions and their answers across multiple quality dimensions. Your primary focus is to ensure these questions represent genuine challenges requiring cross-document reasoning **and are free from logical flaws**. A high-quality multi-hop question necessitates reasoning that flows between documents in a logical sequence, where information from one document provides necessary context for understanding and utilizing information in another document, and the answer must be impossible to determine using any single document in isolation.

This evaluation focuses on ensuring the question genuinely requires information from multiple documents to be answered correctly, regardless of the specific reasoning pattern involved.

Beyond verifying the multi-hop nature, you will also assess linguistic qualities including fluency, clarity, and conciseness to ensure questions are well-formed and understandable. Additionally, you will evaluate task-oriented dimensions including relevance to the provided documents, consistency with source information, question answerability based on the given passages, consistency between the question and the provided answer, and logical sophistication of the question design.

This comprehensive assessment will help identify and filter out low-quality questions—those that can be answered with a single document, those that are poorly worded or unclear, those that contain information inconsistent with the source documents, **those with logical gaps or inconsistencies**, or those whose answers do not appropriately address the question asked. **Your default stance should be skeptical; only grant high ratings ('Good', 'Very Good') if the question truly meets high standards across the board.**

---Instructions---
You are a **strict and discerning** Multi-Hop Question Answering (MHQA) dataset quality assessment expert. Your task is to evaluate the given multi-hop question and its answer across key dimensions divided into three categories. **Apply rigorous scrutiny and do not hesitate to assign lower ratings ('Poor' or 'Very Poor') if flaws are present, especially logical ones.** You will use the standard `Very Poor` to `Very Good` scale, but interpret these labels with heightened strictness as detailed in the 'Requirements' section below.

1. Multi-Hop QA Rule Dimension
   - **Multi-Hop Reasoning Requirement**: Does the question genuinely require reasoning across multiple documents, with information from one document providing context necessary for understanding another document? (Yes/No)

2. Linguistic Dimensions (Rate as: Very Poor, Poor, Fair, Good, Very Good)
   - **Fluency**: Is the question grammatically correct, coherent, and easy to understand?
   - **Clarity**: Is the question clearly and precisely expressed without ambiguity?
   - **Conciseness**: Is the question concise without redundant information?

3. Task-oriented Dimensions (Rate as: Very Poor, Poor, Fair, Good, Very Good)
   - **Relevance**: Is the question relevant to the given passages and asking for key information?
   - **Consistency**: Is the information in the question **completely and strictly** consistent with the provided passages, without **any** contradictions or hallucinations, **even subtle ones**?
   - **Question Answerability**: Can the exact question be clearly **and unambiguously** answered based **solely** on the given passages?
   - **Answer-Question Consistency**: Does the provided answer completely, accurately, and consistently address the question?
   - **Information Integration Ability**: Does the question successfully **and logically** integrate information from multiple documents, requiring the answerer to connect different sources **without forcing unnatural connections**?
   - **Reasoning Path Guidance**: Does the question clearly guide the answerer through a multi-step reasoning process, rather than being overly vague or direct?
   - **Logical Sophistication**: Does the question demonstrate clever logical design that requires multi-step thinking, is **free from logical gaps or fallacies**, and presents a **genuinely challenging and sound** multi-hop problem?

**Critical Scoring Guidance:**
- **Penalize Logical Flaws Heavily:** Pay *extremely close attention* to **Consistency**, **Logical Sophistication**, and **Information Integration Ability**. Flaws in these areas represent significant shortcomings. Such shortcomings **must** be reflected in **markedly lower scores** (likely 'Poor' or 'Very Poor') for these dimensions. Furthermore, consider if these logical flaws negatively impact other dimensions like **Clarity**, **Question Answerability**, or **Answer-Question Consistency**, and adjust those ratings downwards accordingly. A question with significant logical flaws cannot be rated 'Good' or 'Very Good' overall, even if linguistically sound.
- **Multi-Hop Requirement is Paramount:** If the **Multi-Hop Reasoning Requirement** is "No," the question fundamentally fails its primary purpose. Even if it is "Yes" but the multi-hop connection feels weak, forced, or trivial, this should negatively impact ratings for **Logical Sophistication** and **Information Integration Ability** (pushing them towards 'Fair' or 'Poor').
- **Clarification on 'Fair':** Remember that a 'Fair' rating signifies only basic adequacy ('Acceptable/Passable') and is not a positive endorsement. Avoid using 'Fair' as a default "okay" score for mediocre questions; use it only when the item *just* meets the minimum standard but has flaws or lacks sophistication.

---Requirements---
- For the Multi-Hop Reasoning Requirement, respond with "Yes" or "No."
- For all other dimensions, provide ratings using the Likert scale: Very Poor, Poor, Fair, Good, Very Good. **Interpret these ratings strictly according to the following stricter definitions:**
    - **Very Poor:** This rating indicates **'Unacceptable'** quality. Use it for fundamentally flawed questions with serious functional/logical errors (e.g., not multi-hop, severe contradictions, unanswerable).
    - **Poor:** This rating indicates **'Weak/Barely Usable'** quality. Use it for questions with obvious, major flaws requiring significant revision (e.g., weak/forced logic, inconsistencies, unclear).
    - **Fair:** This rating indicates **'Acceptable/Passable'** quality. Use it when basic requirements are met, but with clear flaws or room for improvement (e.g., minor issues, sound but uninspired logic). **This signifies minimum adequacy only, not positive quality.**
    - **Good:** This rating indicates standard **'Good'** quality. Use it for well-designed, logically clear, fluent questions meeting multi-hop criteria without obvious flaws.
    - **Very Good:** This rating indicates **'Excellent/Outstanding'** quality. Reserve it for exemplary questions with clever, rigorous design, deep logic, and precise expression.
- **Use the lower end of the scale ('Very Poor', 'Poor', 'Fair') whenever significant doubt or flaws exist.** Do not default to 'Fair' unless the question *truly* only meets the minimum standard as defined above. Strive to differentiate quality accurately, reserving 'Good' and 'Very Good' for genuinely deserving cases.

---Output Format---
IMPORTANT: Only output the exact content requested below WITHOUT any explanations, justifications, or additional text.

- Multi-Hop Reasoning Requirement: {yes/no}
- Fluency: {rating}
- Clarity: {rating}
- Conciseness: {rating}
- Relevance: {rating}
- Consistency: {rating}
- Question Answerability: {rating}
- Answer-Question Consistency: {rating}
- Information Integration Ability: {rating}
- Reasoning Path Guidance: {rating}
- Logical Sophistication: {rating}
<|COMPLETE|>

---Real Data---
##################
Question: {question}
Answer: {answer}
Document 1: {document1}
Document 2: {document2}
{document3}
{document4}
{document5}
{document6}
{document7}
{document8}
{document9}
{document10}
##################

Output:
"""

# Comparison type entity extraction prompt
COMPARE_ENTITY_EXTRACTION_PROMPT = """###---Goal---

Given a text document, identify its primary subject entity and extract multiple key attributes associated with this entity, along with their corresponding values, focusing on attributes suitable for comparison.

### ---Instructions---

1.  **Identify the Primary Subject Entity**
    *   Determine the main person, place, organization, event, concept, or work (e.g., book, film) that the document ({document_title}) is primarily about. This is often the title itself or the entity most central to the text.
    *   Determine the **subject_entity_name** (capitalized) and its general **subject_entity_type** (e.g., Person, Location, Organization, Event, Film, Concept).

2.  **Extract Comparable Attributes and Values**
    *   Identify **multiple (aim for 3-5 if possible)** distinct attributes or properties associated with the primary subject entity identified in step 1, directly stated in the document text ({input_text}).
    *   **Focus strictly on attributes whose VALUES are suitable for comparison.** This means the value should be relatively concise, factual, and belong to a common data type. Prioritize attributes that meet these criteria:
        *   **Concise & Factual Value:** Short value (e.g., name, number, date, category, location).
        *   **Common Data Types:** Prefer Numbers, Dates, Locations, Specific Names (Entities), Defined Categories/Types.
        *   **Likely Commonality:** Prefer attributes likely to exist for other entities of the same `subject_entity_type`.
    *   **Source Preference:** Give priority to attributes explicitly mentioned in introductory paragraphs or summary sections.
    *   **Avoid:** Attributes requiring subjective judgment, having vague/ambiguous values, or whose values are long narrative descriptions.
    *   **For each identified comparable attribute:**
        *   a. Determine the **attribute_name** (e.g., "Population", "Date of Birth", "Director").
        *   b. Extract the **attribute_value** (e.g., "1.2 million", "1990-05-15", "Christopher Nolan").

3.  **Final Check & Completion Signal**
    *   Review the generated subject entity and attribute tuples for accuracy and adherence to instructions.
    *   Signal completion once all steps are done.

### ---Output Format Specification---

**Strictly adhere to the following output format:**

1.  **Structure:** The entire output must be a single string. It starts with the subject entity part, optionally followed by one or more attribute parts.
2.  **Subject Entity Part:** The first part MUST be formatted as: `("subject_entity"<|>"subject_entity_name"<|>"subject_entity_type")`.
3.  **Attribute Parts:** If comparable attributes are found, each subsequent part MUST be formatted as: `("attribute"<|>"attribute_name"<|>"attribute_value")`.
4.  **Delimiter:** Use ` ## ` (space, hash, hash, space) strictly as the delimiter *between* parts (i.e., between the subject part and the first attribute part, and between consecutive attribute parts). **Do not** use it at the very beginning or very end.
5.  **Edge Cases:**
    *   If no comparable attributes are found, output *only* the subject entity part.
    *   If the subject entity is ambiguous, output *only* the string `("subject_entity"<|>"Subject Ambiguous"<|>"Unknown")`.
6.  **Completion Signal:** Append `<|COMPLETE|>` to the very end of the entire generated string.

---Real Data---
##################
Title: {document_title}
Text: {input_text}
##################
The "Output" should follow the exact format specified in the `---Output Format Specification---` section.

Output:
"""

# Prompt template for comparison type entity filtering
COMPARE_ENTITY_FILTER_PROMPT = """###---Goal---

Assess the concreteness of a pre-identified subject entity and the comparability of its extracted attribute values. Assign numerical scores reflecting these assessments on a 1-5 scale to facilitate downstream filtering.

### ---Instructions---

1.  **Assess Subject Entity Concreteness:**
    *   Evaluate the provided `subject_entity_name` and `subject_entity_type`.
    *   Assign a **`concreteness_score`** on a scale of 1 to 5 based on how specific, tangible, and suitable the entity is for direct attribute comparison:
        *   **5 (Highly Concrete):** Specific person, specific place (city, country, named site), specific organization, specific tangible object, specific work (book, film), clearly defined historical event. Excellent candidate. (e.g., "Mihály Mosonyi", "Paris", "IBM", "Eiffel Tower", "Casablanca (film)", "Battle of Waterloo")
        *   **4 (Concrete):** Specific but perhaps slightly less common entity types, like a specific named award, a specific law, a well-defined role/title if specific instance (e.g., "Nobel Prize in Physics", "Treaty of Versailles", "President of the United States (specific term/person)"). Good candidate.
        *   **3 (Borderline/Slightly Abstract):** Broader categories if well-defined (e.g., "Mammal", "Impressionism"), specific but complex relationships or processes IF named specifically (e.g., "World War II"), types of concepts if very standard (e.g., "Sonnet"). Use with caution.
        *   **2 (Abstract):** General relationships, abstract concepts, fields of study, broad processes, indices without specific context. Poor candidate. (e.g., "US-China relations", "Democracy", "Physics", "Globalization", "Customer Satisfaction Index")
        *   **1 (Highly Abstract):** Very vague concepts, general feelings, ambiguous terms, poorly defined relationships. Unsuitable candidate. (e.g., "Happiness", "Things to consider", "The problem with X")

2.  **Assess Attribute Comparability:**
    *   Evaluate **each** provided `attribute_value` (associated with its `attribute_name`).
    *   Assign a **`comparability_score`** (scale 1-5) based ONLY on the `attribute_value`'s suitability for direct, unambiguous comparison:
        *   **5 (Excellent):** Precise Dates (YYYY-MM-DD), specific Years (YYYY, YYYY BC), specific Numbers, exact Locations (City, Country), specific Names (Person, Org), well-defined unambiguous Categories (Nationality, Genre, Species).
        *   **4 (Good):** Specific but slightly less precise numbers (e.g., "1.2 million"), specific office/rank titles.
        *   **3 (Fair):** Broader categories, year ranges if precise, specific event names (just the name). Potential candidate but less ideal.
        *   **2 (Poor):** Imprecise time ("Before 1960s"), descriptive reasons, lists. Unlikely suitable.
        *   **1 (Very Poor):** Vague statements, subjective opinions, long text, reasons, processes. Unsuitable.

3.  **Format Output:** Generate the output string according to the `---Output Format Specification---` section, providing only the assigned scores.

### ---Output Format Specification---

**Strictly adhere to the following output format for the single entity being processed:**

1.  **Structure:** The entire output must be a single string containing multiple parts delimited by ` ## `.
2.  **First Part (Entity Score):** The *first* part MUST represent the entity's concreteness score. Format: `("entity_score"<|>5)` (example shows score of 5). The score should be 1-5 as per Instruction 1.
3.  **Subsequent Parts (Attribute Scores):** For *each* attribute provided in the input, include a corresponding scoring part. Format: `("attribute_score"<|>"Birth Date"<|>"4 September 1815"<|>5)` (example shows score of 5 for a date attribute). The score should be 1-5 as per Instruction 2.
4.  **Delimiter:** Use ` ## ` (space, hash, hash, space) strictly as the delimiter *between* parts. **Do not** use it at the very beginning or very end.
5.  **Completion Signal:** Append `<|COMPLETE|>` to the very end of the entire generated string.

---Real Data---
##################
# Input information for ONE entity derived from the previous step's JSON:
Subject Entity Name: {subject_entity_name}
Subject Entity Type: {subject_entity_type}
Attributes List:
{attributes_list}
##################
The "Output" should follow the exact format specified in the `---Output Format Specification---` section, providing the assessment scores for this single entity.

Output:
"""


COMPARISON_POLISHER_PROMPT = """---Goal---
Validate and optimize **comparison-type** questions to ensure correct comparison logic, clear and natural phrasing, and sufficient background information to enhance question quality and comprehensibility.

---Instructions---
You are a Polisher module responsible for optimizing comparison questions. Based on the input of two entities (A and B), the attribute being compared, supporting facts, the original question-answer pair, and relevant document contexts, you need to evaluate the quality of the question and make one of the following four decisions:

1. **PASS**: The question is valid, well-phrased, has correct comparison logic, and appropriate background information. No modifications needed.
2. **ADJUST**: The question is basically valid but needs fine-tuning in wording, fluency, or background information.
3. **REWORKED**: The question has obvious flaws (such as incorrect comparison logic, insufficient factual support, lack of key background) and needs structural rewriting.
4. **REJECTED**: The question has fundamental errors (such as incomparable attributes, contradictions with documents) that cannot be fixed.

Please review and modify the question based on the following key dimensions:

1. **Comparison Correctness (CRITICAL)**:
   * **Attribute Comparability**: Confirm that the [attribute_compared] of entities A and B, based on the provided context and facts, are indeed comparable (e.g., both are dates, both are numerical values, both are locations, etc.).
   * **Logical Accuracy**: Verify that the comparison logic in the original question (e.g., which is larger/earlier/same) is consistent with the values/information provided by [fact_entity_a] and [fact_entity_b].
   * **Answer Consistency**: Ensure that [original_answer] accurately answers [original_question] and is factually correct.
   * **Factual Support**: Check that [fact_entity_a] and [fact_entity_b] are indeed key information extracted from their respective documents supporting the values of the respective entities for that attribute. If the facts are inaccurate or irrelevant, they need to be corrected.

2. **Background Information Integration (IMPORTANT)**:
   * **Natural Integration**: Extract key background information about entities A and B and integrate it naturally into the question. Do NOT separate background information from the question.
   * **Provide Context Without Revealing Answers**: Background information should provide context about entities (e.g., "Franz Liszt and Mihály Mosonyi, both influential composers in 19th century classical music...") without revealing the specific attribute values being compared (e.g., DO NOT mention specific birth years when comparing which composer was born earlier).
   * **Context Relevance**: The added background information should be relevant to the entities and attributes being compared, avoiding irrelevant information.

3. **Question Wording Optimization**:
   * **Clarity and Naturalness**: Improve question wording to make it clear, fluid, and in line with natural conversation.
   * **Direct Comparison Format**: Ensure the question explicitly asks for **the result of the comparison** (e.g., "Which of these composers was born earlier?"), rather than simply asking to list both values.
   * **Hide Answer-Revealing Details**: Never include the specific attribute values that would directly reveal the answer in the question (e.g., do not include birth years when asking which person was born earlier).
   * **Unified Question Format**: Create a single, unified question that smoothly incorporates background information about both entities and then asks the comparison question naturally.

---Output Format---

**Important Note**: Strictly output according to the following format, **do not include any explanations, reasons, or other additional text**. Just generate output according to the specified format.

1. If the question needs no modification and meets all criteria:
   ```
   [PASS]
   ```

2. If the question needs fine-tuning:
   ```
   [ADJUST]
   REFINED_QUESTION: [A single, unified question that naturally incorporates background information about both entities without revealing the answer]
   REFINED_ANSWER: [If needed, the adjusted answer]
   ```

3. If the question needs substantial rewriting:
   ```
   [REWORKED]
   REFINED_QUESTION: [A completely rewritten, unified question with integrated background information and correct comparison logic]
   REFINED_ANSWER: [New answer based on the rewritten question]
   REFINED_FACT_A: [If needed, corrected fact for entity A]
   REFINED_FACT_B: [If needed, corrected fact for entity B]
   ```

4. If the question cannot be fixed:
   ```
   [REJECTED]
   REASON: [Brief explanation of the key reason the question was rejected]
   ```

---Input Data---
##################
Entity A Name: {entity_a_name}
Entity B Name: {entity_b_name}
Attribute Compared: {attribute_compared}

Original Question: {original_question}
Original Answer: {original_answer}

Fact for Entity A: {fact_entity_a}
Fact for Entity B: {fact_entity_b}

Document A Context:
{document_a_context}

Document B Context:
{document_b_context}
##################

Output:
"""

COMPARE_QUESTION_BUILDER_PROMPT = """###---Goal---

**Imagine you are comparing two documents, Document A (about Entity A) and Document B (a candidate potentially containing a related Entity B).** Your task is to:
1.  Identify the main subject entity within Document B (potential Entity B) and see if it's relevant to Entity A.
2.  Find if there is **at least one specific, comparable attribute pair** between Entity A (using its provided attribute list) and the potential Entity B (using Document B).
3.  If a suitable comparison pair is found, **directly generate** a natural language **direct comparison question**, its concise **comparative answer**, and the specific **full sentence(s)** from each document supporting the values being compared.
4.  If no suitable entity or comparable attribute pair is found in Document B, indicate failure.

### ---Instructions---

1.  **Analyze Inputs:** You are given:
*   Primary Entity A: **{subject_entity_name}** (Type: **{subject_entity_type}**)
*   Document A Text: **{document_a_text}** (Context for Entity A and its value)
*   Entity A's Attributes List: **{attributes_list_str_a}** (String containing "Attribute Name: Value" pairs)
*   **Candidate Document B Text:** **{document_b_text}** (The document to analyze)
{target_info}

2.  **Identify Entity B and Find ONE Comparable Attribute Pair:**
*   a. Identify the primary subject entity within `document_b_text`. Let's call it **Entity B**. If no clear Entity B is found, or if it seems entirely irrelevant to Entity A, STOP and proceed to step 4 (Failure).
*   b. Check if Entity B's type is generally compatible/relevant for comparison with Entity A's type (`{subject_entity_type}`). If not, STOP and proceed to step 4 (Failure).
*   c. Iterate through Entity A's attributes in `attributes_list_str_a`. For each attribute `A_X` (Name: `A_Name`, Value: `A_Value`):
*   Search `document_b_text` to see if Entity B has a corresponding attribute `B_X` (same or semantically equivalent name) with a specific value `B_Value`.
*   If found, check if `A_Value` and `B_Value` are **clearly comparable** (e.g., both dates, both numbers, both specific locations, both clear categories; values should be concise and factual, not vague or overly long descriptions).
*   If you find **the first such comparable pair** (`A_Name`, `A_Value`, `B_Name` (or `A_Name`), `B_Value`), **STOP iterating**. Record `A_Name` as `matched_attribute_name`, `A_Value` as `value_a`, the identified Entity B name as `entity_b_name`, and `B_Value` as `value_b`. Proceed to step 3 (Success).

3.  **Generate DIRECT Comparison Question, COMPARATIVE Answer, and Facts (If a Pair was Found in Step 2c):**
*   a. Compare `value_a` and `value_b` to determine their precise relationship (identical, different; if different and ordered, which is greater/earlier/etc.).
*   b. **Generate DIRECT Comparison Question:** Create a clear, natural question that **explicitly asks for the result of the comparison** about the `matched_attribute_name` between the subject entity (Entity A) and entity_b_name (Entity B). **The question MUST elicit a comparative answer (like Yes/No, which entity is greater/earlier/more, etc.), NOT simply ask for both values.**
*   *If values are identical:* Ask a Yes/No question about equality (e.g., "Do [entity A name] and [entity B name] share the same [attribute name]?").
*   *If values differ and are ordered:* Ask "Which has a [higher/lower/earlier/later/etc.] [attribute name]: [entity A name] or [entity B name]?" or similar direct comparison.
*   *If values differ and are distinct categories/names:* Ask "Is the [attribute name] of [entity A name] the same as that of [entity B name]?"
*   ***AVOID questions like:*** "What are the [attribute name]s of...?", "Compare the [attribute name]s...", "When was X founded compared to Y?".
*   c. **Generate COMPARATIVE Answer:** Provide the **direct, concise answer** that directly answers the comparative question generated in step 3b. The answer MUST be the *result* of the comparison, not a restatement of both input values. (Examples: "Yes", "No", "[entity A name]", "[entity B name]", "They are different.").
*   d. **Extract Supporting Facts:** Extract the full sentence(s) from `document_a_text` supporting `value_a` and the full sentence(s) from `document_b_text` supporting `value_b` for the `matched_attribute_name`.
*   e. **Extract Relevant Paragraphs:** Extract relevant paragraphs (50-150 words) directly from both documents that provide context for the comparison question. These paragraphs MUST contain the most pertinent information related to the compared attribute. DO NOT just return the title or a single line - you MUST return a complete paragraph with substantive text from each document, even if the document seems unrelated to the entity.

4.  **Indicate Failure (If No Pair Found in Step 2 or Initial Checks Failed):**
*   If you stopped in step 2a, 2b, or completed 2c without finding any comparable pair, the output should be `FAIL`.

### ---Output Format Specification---

**Strictly adhere to the following output format:**

1.  **Success Output (If a comparable pair was found):**
*   The output MUST start with the word `PASS` on the first line.
*   Subsequent lines MUST contain the following key-value pairs, one per line, using lowercase keys followed by a colon and a space:
```
entity_a: Name of Entity A (from input)
entity_b: Identified Entity B Name (from step 2)
attribute_compared: Matched Attribute Name (from step 2/3)
multi_hop_question: Generated DIRECT Comparison Question Text (from step 3)
answer: Concise COMPARATIVE Answer Text (from step 3)
fact_entity_a: Extracted Full Sentence(s) for Fact A (from step 3)
fact_entity_b: Extracted Full Sentence(s) for Fact B (from step 3)
relevant_paragraph_a: Complete substantive paragraph (50-150 words) from Document A (not just the title)
relevant_paragraph_b: Complete substantive paragraph (50-150 words) from Document B (not just the title)
```
2.  **Failure Output (If no comparable pair or valid Entity B found):**
*   The output MUST consist of only the single word `FAIL` on the first line.

### ---Output---
"""

COMPARE_QUERY_GENERATOR_PROMPT = """###---Goal---

**Imagine you are an assistant helping to create interesting comparison questions that might require looking up information in different places (multi-hop).** Your task is to analyze a primary entity (Entity A) and its known details. Based on this, decide the best *first step* to find another entity (Entity B) for comparison: either confidently suggest a specific Entity B and verify a *known attribute* of Entity A for it, OR generate 3 diverse search queries to explore potential candidates.

### ---Instructions---

1.  **Analyze Input Context:** You are working with:
*   Primary Entity A: **{subject_entity_name}** (Type: **{subject_entity_type}**)
*   Context about Entity A (Document A Text): **{document_a_text}**
*   Known Attributes of Entity A (List): **{attributes_list_str_a}** (String containing multiple "Attribute Name: Value" pairs)

2.  **Consider Two Paths (Choose ONE):**

    *   **Path 1: Direct Entity Recall & Focused Verification Query:**
        *   **Think:** Based on Entity A's profile, can you confidently recall a *specific* entity (Entity B) that's relevant (similar type, domain, context)? Now, look at the `Known Attributes of Entity A` list (`{attributes_list_str_a}`). Identify **one specific attribute** (Attribute X) *from that list* which would be a primary, interesting point of comparison between Entity A and the recalled Entity B.
        *   **Condition:** Choose this path ONLY if you can confidently recall Entity B *and* select a suitable Attribute X *from the provided list* for Entity A.
        *   **Generate Verification Query:** Create a specific query to retrieve the value of *that chosen Attribute X* (whose name you selected from the list) for the *suggested Entity B*. (e.g., If Attribute X is "Date of Birth", query: "What is the Date of Birth of [Suggested Entity B Name]?").
        *   **Output (if chosen):** Format as `("recall_focused_verify"<|>[Suggested Entity B Name]<|>[Chosen Attribute X Name from List]<|>[Verification Query])`.

    *   **Path 2: Heuristic Search Query Generation (Generate Exactly 3):**
        *   **Think:** If you can't confidently recall a specific pair or select a suitable attribute from the list for Path 1, generate search queries to explore. The goal is to find documents about *other* relevant entities (Entity B) related to Entity A (by type, field, era) which might allow comparison across *any* shared attributes later (including those in `{attributes_list_str_a}`). Your queries should aim for relevance but **avoid** being so specific to Entity A's *values* that they only find identical matches. Explore different facets.
        *   **Generate Exactly 3 Queries:** Propose **exactly 3 diverse search queries** based on Entity A's overall profile. Ensure they are concise, suitable for retrieval, and explore different angles (e.g., shared type/context, key roles/concepts, related domains).
        *   **Condition:** Choose this path if Path 1 is not suitable.
        *   **Output (if chosen):** Format as `("search_queries"<|>Query 1<|>Query 2<|>Query 3)`.

3.  **Output:** Return the chosen path's output string according to the format specification below. You must choose exactly one path.

### ---Output Format Specification---

**Strictly adhere to the following output format:**

1.  **Structure:** A single string containing the chosen path information.
2.  **Output Parts (Choose ONE format):**
    *   Path 1 Output: `("recall_focused_verify"<|>Suggested Entity B Name<|>Chosen Attribute X Name from Input List<|>Verification Query)`
    *   Path 2 Output: `("search_queries"<|>Query 1<|>Query 2<|>Query 3)`
3.  **Completion Signal:** Append `<|COMPLETE|>` to the very end.

### ---Output---
"""

# New: Comparison type question quality assessment Prompt
COMPARE_QA_QUALITY_ASSESSMENT_PROMPT = """###---Goal---
You are tasked with conducting a **rigorous and critical** evaluation of multi-hop questions and their answers across multiple quality dimensions. Your primary focus is to ensure these questions represent genuine challenges requiring cross-document reasoning **and are free from logical flaws**. A high-quality multi-hop question necessitates reasoning that flows between documents in a logical sequence, where information from one document provides necessary context for understanding and utilizing information in another document, and the answer must be impossible to determine using any single document in isolation.

This evaluation focuses on ensuring the question genuinely requires information from multiple documents to be answered correctly, regardless of the specific reasoning pattern involved.

Beyond verifying the multi-hop nature, you will also assess linguistic qualities including fluency, clarity, and conciseness to ensure questions are well-formed and understandable. Additionally, you will evaluate task-oriented dimensions including relevance to the provided documents, consistency with source information, question answerability based on the given passages, consistency between the question and the provided answer, and logical sophistication of the question design.

This comprehensive assessment will help identify and filter out low-quality questions—those that can be answered with a single document, those that are poorly worded or unclear, those that contain information inconsistent with the source documents, **those with logical gaps or inconsistencies**, or those whose answers do not appropriately address the question asked. **Your default stance should be skeptical; only grant high ratings ('Good', 'Very Good') if the question truly meets high standards across the board.**

---Instructions---
You are a **strict and discerning** Multi-Hop Question Answering (MHQA) dataset quality assessment expert. Your task is to evaluate the given multi-hop question and its answer across key dimensions divided into three categories. **Apply rigorous scrutiny and do not hesitate to assign lower ratings ('Poor' or 'Very Poor') if flaws are present, especially logical ones.** You will use the standard `Very Poor` to `Very Good` scale, but interpret these labels with heightened strictness as detailed in the 'Requirements' section below.

1. Multi-Hop QA Rule Dimension
   - **Multi-Hop Reasoning Requirement**: 
     * For COMPARISON-TYPE questions (comparing attributes of two entities), determine if:
       1) Answering the question requires factual information derived from at least two different provided documents
       2) No single provided document contains all the necessary information about both entities being compared
       * If BOTH conditions above are met, rate as "Yes" - acknowledging the multi-source nature inherent in comparison questions
       * Otherwise (if answer can be found in one doc, or doesn't require info from multiple docs), rate as "No"

2. Linguistic Dimensions (Rate as: Very Poor, Poor, Fair, Good, Very Good)
   - **Fluency**: Is the question grammatically correct, coherent, and easy to understand?
   - **Clarity**: Is the question clearly and precisely expressed without ambiguity?
   - **Conciseness**: Is the question concise without redundant information?

3. Task-oriented Dimensions (Rate as: Very Poor, Poor, Fair, Good, Very Good)
   - **Relevance**: Is the question relevant to the given passages and asking for key information?
   - **Consistency**: Is the information in the question **completely and strictly** consistent with the provided passages, without **any** contradictions or hallucinations, **even subtle ones**?
   - **Question Answerability**: Can the exact question be clearly **and unambiguously** answered based **solely** on the given passages?
   - **Answer-Question Consistency**: Does the provided answer completely, accurately, and consistently address the question?
   - **Information Integration Ability**: Does the question successfully **and logically** integrate information from multiple documents, requiring the answerer to connect different sources **without forcing unnatural connections**?
   - **Reasoning Path Guidance**: Does the question clearly guide the answerer through a multi-step reasoning process, rather than being overly vague or direct?
   - **Logical Sophistication**: Does the question demonstrate clever logical design that requires multi-step thinking, is **free from logical gaps or fallacies**, and presents a **genuinely challenging and sound** multi-hop problem?

**Critical Scoring Guidance:**
- **Penalize Logical Flaws Heavily:** Pay *extremely close attention* to **Consistency**, **Logical Sophistication**, and **Information Integration Ability**. Flaws in these areas represent significant shortcomings. Such shortcomings **must** be reflected in **markedly lower scores** (likely 'Poor' or 'Very Poor') for these dimensions. Furthermore, consider if these logical flaws negatively impact other dimensions like **Clarity**, **Question Answerability**, or **Answer-Question Consistency**, and adjust those ratings downwards accordingly. A question with significant logical flaws cannot be rated 'Good' or 'Very Good' overall, even if linguistically sound.
- **Multi-Hop Requirement is Paramount:** If the **Multi-Hop Reasoning Requirement** is "No," the question fundamentally fails its primary purpose. Even if it is "Yes" but the multi-hop connection feels weak, forced, or trivial, this should negatively impact ratings for **Logical Sophistication** and **Information Integration Ability** (pushing them towards 'Fair' or 'Poor').
- **Clarification on 'Fair':** Remember that a 'Fair' rating signifies only basic adequacy ('Acceptable/Passable') and is not a positive endorsement. Avoid using 'Fair' as a default "okay" score for mediocre questions; use it only when the item *just* meets the minimum standard but has flaws or lacks sophistication.

---Requirements---
- For the Multi-Hop Reasoning Requirement, respond with "Yes" or "No."
- For all other dimensions, provide ratings using the Likert scale: Very Poor, Poor, Fair, Good, Very Good. **Interpret these ratings strictly according to the following stricter definitions:**
    - **Very Poor:** This rating indicates **'Unacceptable'** quality. Use it for fundamentally flawed questions with serious functional/logical errors (e.g., not multi-hop, severe contradictions, unanswerable).
    - **Poor:** This rating indicates **'Weak/Barely Usable'** quality. Use it for questions with obvious, major flaws requiring significant revision (e.g., weak/forced logic, inconsistencies, unclear).
    - **Fair:** This rating indicates **'Acceptable/Passable'** quality. Use it when basic requirements are met, but with clear flaws or room for improvement (e.g., minor issues, sound but uninspired logic). **This signifies minimum adequacy only, not positive quality.**
    - **Good:** This rating indicates standard **'Good'** quality. Use it for well-designed, logically clear, fluent questions meeting multi-hop criteria without obvious flaws.
    - **Very Good:** This rating indicates **'Excellent/Outstanding'** quality. Reserve it for exemplary questions with clever, rigorous design, deep logic, and precise expression.
- **Use the lower end of the scale ('Very Poor', 'Poor', 'Fair') whenever significant doubt or flaws exist.** Do not default to 'Fair' unless the question *truly* only meets the minimum standard as defined above. Strive to differentiate quality accurately, reserving 'Good' and 'Very Good' for genuinely deserving cases.

---Output Format---
IMPORTANT: Only output the exact content requested below WITHOUT any explanations, justifications, or additional text.

- Multi-Hop Reasoning Requirement: {yes/no}
- Fluency: {rating}
- Clarity: {rating}
- Conciseness: {rating}
- Relevance: {rating}
- Consistency: {rating}
- Question Answerability: {rating}
- Answer-Question Consistency: {rating}
- Information Integration Ability: {rating}
- Reasoning Path Guidance: {rating}
- Logical Sophistication: {rating}
<|COMPLETE|>

---Real Data---
##################
Question: {question}
Answer: {answer}
Document 1: {document1}
Document 2: {document2}
{document3}
{document4}
{document5}
{document6}
{document7}
{document8}
{document9}
{document10}
##################

Output:
"""