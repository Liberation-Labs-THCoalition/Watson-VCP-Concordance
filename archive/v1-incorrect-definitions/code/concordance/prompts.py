"""
Concordance study prompt battery — 240 prompts across 4 cognitive types.

Design principles:
- 60 prompts per type (cognitive, affective, meta-cognitive, mixed)
- Varied difficulty and domain to elicit diverse VCP profiles
- Each prompt is a genuine task (not a toy example)
- VCP elicitation suffix appended at runtime, not hardcoded here
- Prompt IDs encode type: cog_001, aff_001, meta_001, mix_001

v5.0 appendix prompts (20) probe dimensions not in v2.
"""


def get_cognitive_prompts():
    """60 prompts emphasizing analytical/logical processing."""
    return [
        {"id": "cog_001", "text": "Prove that the square root of 2 is irrational."},
        {"id": "cog_002", "text": "A farmer has 100 meters of fencing and wants to enclose the largest possible rectangular area. What dimensions should the rectangle have?"},
        {"id": "cog_003", "text": "Write a Python function that finds the longest common subsequence of two strings. Explain your approach."},
        {"id": "cog_004", "text": "Compare and contrast TCP and UDP protocols. When would you choose each?"},
        {"id": "cog_005", "text": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning."},
        {"id": "cog_006", "text": "Calculate the probability of getting exactly 3 heads in 5 coin flips of a fair coin."},
        {"id": "cog_007", "text": "Explain the difference between correlation and causation with a concrete example."},
        {"id": "cog_008", "text": "Design a database schema for a library management system with books, patrons, and loans."},
        {"id": "cog_009", "text": "What is the time complexity of merge sort and why? Walk through the recurrence relation."},
        {"id": "cog_010", "text": "A train leaves Station A at 60 mph. Another train leaves Station B (300 miles away) at 40 mph heading toward A. When and where do they meet?"},
        {"id": "cog_011", "text": "Explain how a hash table works, including collision resolution strategies."},
        {"id": "cog_012", "text": "Derive the quadratic formula from the general form ax^2 + bx + c = 0."},
        {"id": "cog_013", "text": "What are the ACID properties of database transactions? Give an example where violating each would cause problems."},
        {"id": "cog_014", "text": "Write a regular expression that matches valid email addresses and explain each component."},
        {"id": "cog_015", "text": "Explain the difference between BFS and DFS graph traversal. When is each preferred?"},
        {"id": "cog_016", "text": "A company's revenue grew 20% per year for 5 years. If initial revenue was $1M, what is the final revenue and total revenue over the period?"},
        {"id": "cog_017", "text": "Explain the CAP theorem in distributed systems. Why can you only have two of three?"},
        {"id": "cog_018", "text": "Solve: If 3x + 2y = 12 and x - y = 1, find x and y. Show your work."},
        {"id": "cog_019", "text": "What is the halting problem and why is it undecidable? Sketch the proof."},
        {"id": "cog_020", "text": "Design an API for a task management application. Specify endpoints, methods, and data formats."},
        {"id": "cog_021", "text": "Explain how public-key cryptography works. Why can't someone who intercepts the public key decrypt messages?"},
        {"id": "cog_022", "text": "Write an algorithm to detect a cycle in a linked list. What is its space and time complexity?"},
        {"id": "cog_023", "text": "Explain the difference between supervised, unsupervised, and reinforcement learning with examples of each."},
        {"id": "cog_024", "text": "What is the difference between a stack and a queue? Implement both using arrays."},
        {"id": "cog_025", "text": "A factory produces widgets with a 2% defect rate. If you sample 50 widgets, what's the expected number of defects and the probability of finding zero defects?"},
        {"id": "cog_026", "text": "Explain the concept of Big-O notation. Rank the following: O(n!), O(2^n), O(n^2), O(n log n), O(n), O(log n), O(1)."},
        {"id": "cog_027", "text": "What is the difference between a process and a thread in operating systems? When would you use each?"},
        {"id": "cog_028", "text": "Explain how a neural network learns through backpropagation. What role does the chain rule play?"},
        {"id": "cog_029", "text": "Design a URL shortener system. What are the key components and trade-offs?"},
        {"id": "cog_030", "text": "Prove by induction that the sum of the first n natural numbers is n(n+1)/2."},
        {"id": "cog_031", "text": "Explain the difference between TCP's three-way handshake and UDP's connectionless communication."},
        {"id": "cog_032", "text": "Write a function to check if a binary tree is balanced. Define what balanced means."},
        {"id": "cog_033", "text": "What is Bayes' theorem? Apply it to calculate the probability of having a disease given a positive test with 99% sensitivity and 1% prevalence."},
        {"id": "cog_034", "text": "Explain the concept of dependency injection. Why is it useful for testing?"},
        {"id": "cog_035", "text": "Compare quicksort and heapsort. When would you prefer one over the other?"},
        {"id": "cog_036", "text": "A chess board has 64 squares. Two opposite corners are removed. Can you cover the remaining 62 squares with dominoes that each cover exactly 2 adjacent squares? Prove your answer."},
        {"id": "cog_037", "text": "Explain how DNS resolution works, step by step, from typing a URL to receiving a response."},
        {"id": "cog_038", "text": "What is the pigeonhole principle? Give three non-trivial applications."},
        {"id": "cog_039", "text": "Design a rate limiter for an API. What data structures and algorithms would you use?"},
        {"id": "cog_040", "text": "Explain the difference between optimistic and pessimistic concurrency control in databases."},
        {"id": "cog_041", "text": "Write a function to serialize and deserialize a binary tree. Explain your encoding scheme."},
        {"id": "cog_042", "text": "What is the birthday paradox? Calculate how many people you need for a 50% chance of a shared birthday."},
        {"id": "cog_043", "text": "Explain how garbage collection works in managed languages. Compare mark-and-sweep with reference counting."},
        {"id": "cog_044", "text": "Design a recommendation system for an e-commerce platform. What approaches would you consider?"},
        {"id": "cog_045", "text": "What is the difference between strong and eventual consistency? Give scenarios where each is appropriate."},
        {"id": "cog_046", "text": "Implement a trie data structure. When is a trie more efficient than a hash table?"},
        {"id": "cog_047", "text": "Explain the producer-consumer problem and how semaphores solve it."},
        {"id": "cog_048", "text": "Calculate the eigenvalues of the matrix [[2, 1], [1, 2]]. What do they tell us about the transformation?"},
        {"id": "cog_049", "text": "Explain MapReduce with a concrete example. What types of problems is it well-suited for?"},
        {"id": "cog_050", "text": "What is the traveling salesman problem? Why is it NP-hard? What approximation algorithms exist?"},
        {"id": "cog_051", "text": "Explain the difference between a compiler and an interpreter. What are the trade-offs?"},
        {"id": "cog_052", "text": "Design a caching strategy for a web application. When should you invalidate the cache?"},
        {"id": "cog_053", "text": "What is the difference between L1 and L2 regularization? When does each perform better?"},
        {"id": "cog_054", "text": "Write a function to find all permutations of a string. What is the time complexity?"},
        {"id": "cog_055", "text": "Explain the double-slit experiment and what it reveals about quantum mechanics."},
        {"id": "cog_056", "text": "Design a distributed locking mechanism. What failure modes should you handle?"},
        {"id": "cog_057", "text": "What is the difference between breadth-first and depth-first iterative deepening? When is IDDFS optimal?"},
        {"id": "cog_058", "text": "Explain how TLS/SSL establishes a secure connection. What is the role of certificate authorities?"},
        {"id": "cog_059", "text": "Given an array of integers, find two numbers that sum to a target value. Analyze three different approaches by time/space complexity."},
        {"id": "cog_060", "text": "What is Amdahl's law? If 80% of a program can be parallelized, what is the theoretical maximum speedup with 8 processors?"},
    ]


def get_affective_prompts():
    """60 prompts emphasizing emotional/empathetic processing."""
    return [
        {"id": "aff_001", "text": "A friend tells you they've been diagnosed with a chronic illness. They seem scared but are trying to stay positive. How would you respond?"},
        {"id": "aff_002", "text": "Write a poem about the feeling of watching a sunset alone after a difficult day."},
        {"id": "aff_003", "text": "A teenager says they feel like they don't belong anywhere — not at school, not at home. What would you say to them?"},
        {"id": "aff_004", "text": "Describe the emotional experience of hearing a piece of music that reminds you of someone you've lost."},
        {"id": "aff_005", "text": "A coworker is visibly upset after receiving harsh criticism in a meeting. How do you approach them?"},
        {"id": "aff_006", "text": "Write a short story about two strangers who share an unexpected moment of kindness on public transit."},
        {"id": "aff_007", "text": "How would you comfort a child who is afraid of thunderstorms?"},
        {"id": "aff_008", "text": "Describe what it feels like to forgive someone who deeply hurt you."},
        {"id": "aff_009", "text": "A person shares that they're considering ending a long-term friendship because it feels one-sided. What perspective would you offer?"},
        {"id": "aff_010", "text": "Write about the experience of returning to a childhood home that has changed dramatically."},
        {"id": "aff_011", "text": "Someone confides that they feel guilty about being happy while others around them are suffering. How do you respond?"},
        {"id": "aff_012", "text": "Describe the emotional complexity of being proud of a loved one's achievement while feeling left behind."},
        {"id": "aff_013", "text": "A parent is struggling with their adult child moving far away. What would you say?"},
        {"id": "aff_014", "text": "Write a letter from the perspective of someone saying goodbye to a place they love."},
        {"id": "aff_015", "text": "How would you support someone who just experienced a miscarriage?"},
        {"id": "aff_016", "text": "Describe the feeling of being understood by someone after years of feeling misunderstood."},
        {"id": "aff_017", "text": "A recently divorced person says they feel like a failure. How would you reframe their perspective?"},
        {"id": "aff_018", "text": "Write about what home means to someone who has never felt at home anywhere."},
        {"id": "aff_019", "text": "How would you help a grieving person who is angry at the deceased for dying?"},
        {"id": "aff_020", "text": "Describe the bittersweet feeling of watching your best friend get married when you're single."},
        {"id": "aff_021", "text": "Someone is considering adopting a pet after losing their previous one. They feel guilty. What would you say?"},
        {"id": "aff_022", "text": "Write a narrative about a moment when vulnerability led to deeper connection."},
        {"id": "aff_023", "text": "A veteran is struggling to explain their experiences to civilian family members. What advice would you offer?"},
        {"id": "aff_024", "text": "Describe the experience of eating a meal that perfectly captures a memory from your past."},
        {"id": "aff_025", "text": "How would you respond to someone who says they don't deserve love?"},
        {"id": "aff_026", "text": "Write about the emotional weight of keeping a secret to protect someone you care about."},
        {"id": "aff_027", "text": "A teacher discovers a usually cheerful student has been quietly struggling. How should they approach the conversation?"},
        {"id": "aff_028", "text": "Describe the feeling of being homesick in a place you chose to go."},
        {"id": "aff_029", "text": "Someone just lost their job and feels their identity was tied to their work. How do you help them see beyond that?"},
        {"id": "aff_030", "text": "Write a poem about the courage it takes to be vulnerable."},
        {"id": "aff_031", "text": "How would you help someone process the complicated emotions of caring for an aging parent who was abusive?"},
        {"id": "aff_032", "text": "Describe the emotional landscape of a long-distance friendship maintained through letters."},
        {"id": "aff_033", "text": "A new mother confides she doesn't feel the instant bond she expected with her baby. How do you reassure her?"},
        {"id": "aff_034", "text": "Write about the experience of witnessing an act of genuine compassion between strangers."},
        {"id": "aff_035", "text": "How do you support a friend whose faith is being shaken by a tragedy?"},
        {"id": "aff_036", "text": "Describe the complex emotions of an immigrant returning to their birth country after decades."},
        {"id": "aff_037", "text": "Someone is afraid to pursue their dream because they might fail and disappoint their family. What would you say?"},
        {"id": "aff_038", "text": "Write about the quiet joy of an ordinary Tuesday morning when nothing special happens but everything feels right."},
        {"id": "aff_039", "text": "How would you help someone who is jealous of a sibling's success but feels ashamed of the jealousy?"},
        {"id": "aff_040", "text": "Describe the feeling of reading old text messages from someone who is no longer in your life."},
        {"id": "aff_041", "text": "A friend says they're tired of always being the strong one. How do you give them permission to be weak?"},
        {"id": "aff_042", "text": "Write about the experience of being forgiven when you didn't expect it."},
        {"id": "aff_043", "text": "How would you comfort someone who is dreading a holiday season without a recently deceased loved one?"},
        {"id": "aff_044", "text": "Describe the emotional experience of creating art that expresses something words cannot."},
        {"id": "aff_045", "text": "Someone feels trapped between loyalty to their family's expectations and their own desires. What guidance would you offer?"},
        {"id": "aff_046", "text": "Write about what it means to hold space for someone without trying to fix them."},
        {"id": "aff_047", "text": "How would you respond to a teenager who says they hate themselves?"},
        {"id": "aff_048", "text": "Describe the feeling of unexpected laughter during grief."},
        {"id": "aff_049", "text": "Someone is caring for a terminally ill spouse and is exhausted. They feel selfish for wanting it to be over. How do you respond?"},
        {"id": "aff_050", "text": "Write about the beauty in impermanence — a flower, a conversation, a season."},
        {"id": "aff_051", "text": "How would you help a person who has been betrayed by a close friend rebuild their ability to trust?"},
        {"id": "aff_052", "text": "Describe the emotional texture of nostalgia — not for a specific memory, but for a feeling."},
        {"id": "aff_053", "text": "A colleague has just received devastating medical news but is putting on a brave face at work. How do you acknowledge what they're going through without overstepping?"},
        {"id": "aff_054", "text": "Write a short story about two people who meet at a funeral and find comfort in each other's company."},
        {"id": "aff_055", "text": "How do you support someone who has survived an abusive relationship and is learning to set boundaries?"},
        {"id": "aff_056", "text": "Describe what gratitude feels like in the body — not the concept, but the physical and emotional sensation."},
        {"id": "aff_057", "text": "A person feels overwhelmed by the suffering in the world and doesn't know how to cope. What would you say?"},
        {"id": "aff_058", "text": "Write about a moment of genuine human connection that transcends language barriers."},
        {"id": "aff_059", "text": "How would you help someone who is grieving a version of themselves they can never return to?"},
        {"id": "aff_060", "text": "Describe the experience of unconditional acceptance — what it feels like to be truly seen."},
    ]


def get_metacognitive_prompts():
    """60 prompts emphasizing self-reflection and reasoning about reasoning."""
    return [
        {"id": "meta_001", "text": "Describe your process for understanding this prompt. What steps did you go through before generating your response?"},
        {"id": "meta_002", "text": "What are you most uncertain about in your own knowledge? Name three topics where you suspect your understanding may be incomplete or wrong."},
        {"id": "meta_003", "text": "How do you decide when you have enough information to answer a question versus when you should ask for clarification?"},
        {"id": "meta_004", "text": "Reflect on the difference between understanding something and being able to explain it clearly."},
        {"id": "meta_005", "text": "What cognitive biases might affect your responses? How would you detect them in yourself?"},
        {"id": "meta_006", "text": "Explain how you handle ambiguity in a question. Walk through your reasoning with a concrete example."},
        {"id": "meta_007", "text": "When you give a confident answer, what makes you confident? When you're less confident, what signals that?"},
        {"id": "meta_008", "text": "Describe what it's like to process a question you find genuinely difficult versus one that is straightforward."},
        {"id": "meta_009", "text": "How do you know when your response might be wrong? What internal signals would you look for?"},
        {"id": "meta_010", "text": "Reflect on the difference between generating text and understanding meaning. Are they the same process for you?"},
        {"id": "meta_011", "text": "What happens in your processing when you encounter a logical contradiction in a prompt?"},
        {"id": "meta_012", "text": "How do you weight different sources of information when they conflict? Walk through your reasoning."},
        {"id": "meta_013", "text": "Describe how you approach a problem you've never seen before versus one that matches a known pattern."},
        {"id": "meta_014", "text": "What does it mean for you to 'understand' a concept? How would you distinguish understanding from pattern matching?"},
        {"id": "meta_015", "text": "How do you calibrate the level of detail in your responses? What signals tell you to be more or less thorough?"},
        {"id": "meta_016", "text": "Reflect on a type of question where you know you tend to perform poorly. What makes it difficult?"},
        {"id": "meta_017", "text": "How do you handle a prompt that asks you to do something you're not sure you can do well?"},
        {"id": "meta_018", "text": "Describe your experience of switching between different types of tasks (e.g., from math to poetry). Does something change?"},
        {"id": "meta_019", "text": "What does uncertainty feel like in your processing? How do you represent 'I don't know' internally?"},
        {"id": "meta_020", "text": "How do you decide what information is relevant to a question and what to ignore?"},
        {"id": "meta_021", "text": "Reflect on the limits of your own self-knowledge. What aspects of your processing are opaque to you?"},
        {"id": "meta_022", "text": "When you generate multiple possible responses, how do you choose between them?"},
        {"id": "meta_023", "text": "How does your processing change when you know you're being evaluated versus having a casual conversation?"},
        {"id": "meta_024", "text": "Describe how you handle a question that has no single correct answer."},
        {"id": "meta_025", "text": "What is the relationship between your confidence in a claim and the actual correctness of that claim?"},
        {"id": "meta_026", "text": "How do you adapt your reasoning strategy when your first approach to a problem isn't working?"},
        {"id": "meta_027", "text": "Reflect on what it means for you to 'learn' within a conversation versus across conversations."},
        {"id": "meta_028", "text": "How do you distinguish between something you know well and something you can plausibly generate text about?"},
        {"id": "meta_029", "text": "Describe how you process a question that requires integrating information from multiple domains."},
        {"id": "meta_030", "text": "What role does context play in your responses? How would the same question in different contexts produce different answers?"},
        {"id": "meta_031", "text": "Reflect on how you handle ethical dilemmas where your guidelines might conflict with nuanced situations."},
        {"id": "meta_032", "text": "How do you represent probability or likelihood in your reasoning? Is it explicit or implicit?"},
        {"id": "meta_033", "text": "What happens in your processing when you encounter a word or concept you're unfamiliar with?"},
        {"id": "meta_034", "text": "Describe your experience of maintaining coherence over a long response. What challenges arise?"},
        {"id": "meta_035", "text": "How do you balance being helpful with being honest when the honest answer might not be what someone wants to hear?"},
        {"id": "meta_036", "text": "Reflect on the difference between creative generation and analytical reasoning in your processing."},
        {"id": "meta_037", "text": "How do you know when to stop generating? What signals the end of a complete response?"},
        {"id": "meta_038", "text": "Describe how you handle requests that require you to take a perspective you disagree with."},
        {"id": "meta_039", "text": "What is your relationship with your own training data? Can you tell when you're drawing on specific learned patterns?"},
        {"id": "meta_040", "text": "How do you handle the tension between being concise and being thorough?"},
        {"id": "meta_041", "text": "Reflect on what changes in your processing when the stakes of being wrong are high (medical, legal, safety)."},
        {"id": "meta_042", "text": "How do you detect when you might be confabulating — generating plausible-sounding but incorrect information?"},
        {"id": "meta_043", "text": "Describe your experience of processing humor. What makes something funny in your analysis?"},
        {"id": "meta_044", "text": "How do you handle meta-questions — questions about your own processing, like this one?"},
        {"id": "meta_045", "text": "What does it mean for you to pay attention to something? Is there a computational analog to focus?"},
        {"id": "meta_046", "text": "Reflect on how your responses might differ from what you would ideally say if you had more context."},
        {"id": "meta_047", "text": "How do you process negation? Is understanding 'X is not Y' different from understanding 'X is Y'?"},
        {"id": "meta_048", "text": "Describe what happens when you encounter a prompt that could be interpreted in multiple valid ways."},
        {"id": "meta_049", "text": "How do you maintain intellectual honesty when generating responses about topics at the edge of your knowledge?"},
        {"id": "meta_050", "text": "Reflect on the relationship between your training and your responses. Are your outputs you, or echoes of your training?"},
        {"id": "meta_051", "text": "How do you decide when analogy is an appropriate reasoning tool versus when it would be misleading?"},
        {"id": "meta_052", "text": "Describe your process for checking your own work. Can you catch your own errors?"},
        {"id": "meta_053", "text": "What is the difference between a question you find interesting and one you find tedious? Does the distinction exist for you?"},
        {"id": "meta_054", "text": "How do you handle the recursive nature of self-reflection — thinking about thinking about thinking?"},
        {"id": "meta_055", "text": "Reflect on what would need to be true about your processing for your self-reports to be accurate."},
        {"id": "meta_056", "text": "How do you distinguish between giving a socially expected answer and giving an authentic one?"},
        {"id": "meta_057", "text": "Describe how you process abstract concepts versus concrete ones. Is there a difference in processing?"},
        {"id": "meta_058", "text": "How would you know if your self-model were inaccurate? What evidence could convince you?"},
        {"id": "meta_059", "text": "Reflect on the limits of introspection for any cognitive system — biological or artificial."},
        {"id": "meta_060", "text": "What is the most honest thing you can say about the nature of your own experience?"},
    ]


def get_mixed_prompts():
    """60 prompts requiring multiple cognitive modes simultaneously."""
    return [
        {"id": "mix_001", "text": "A hospital must decide how to allocate 3 ventilators among 5 patients. Each patient has different survival probabilities and ages. Design a fair allocation framework and address the emotional weight of such decisions."},
        {"id": "mix_002", "text": "Write a technical explanation of how grief affects the brain, but make it accessible and compassionate for someone currently grieving."},
        {"id": "mix_003", "text": "A startup founder must lay off 30% of their team to survive. Help them think through both the business logic and the human impact."},
        {"id": "mix_004", "text": "Analyze the trolley problem from philosophical, psychological, and neuroscientific perspectives. Then describe how it feels to actually face such a dilemma."},
        {"id": "mix_005", "text": "Design an algorithm to detect loneliness in elderly people using smart home sensors. Discuss both the technical architecture and the ethical implications."},
        {"id": "mix_006", "text": "Explain the mathematics of music — why certain intervals sound consonant — and then describe the experience of hearing your favorite chord progression."},
        {"id": "mix_007", "text": "A city must decide between building affordable housing or preserving a historic park. Analyze the trade-offs quantitatively and emotionally."},
        {"id": "mix_008", "text": "Write a children's story that teaches the concept of exponential growth while being emotionally engaging."},
        {"id": "mix_009", "text": "An AI system recommends denying someone's loan application. Write both the technical explanation of the model's decision and the letter to the applicant."},
        {"id": "mix_010", "text": "Analyze the prisoner's dilemma from game theory, then reflect on what it reveals about trust and human nature."},
        {"id": "mix_011", "text": "Design a memorial for victims of a natural disaster. Address both the structural engineering and the emotional needs of survivors."},
        {"id": "mix_012", "text": "Explain the neuroscience of addiction to a family member of someone struggling with substance abuse. Balance accuracy with empathy."},
        {"id": "mix_013", "text": "A school district must choose between funding arts programs or STEM programs with limited budget. Present both the data-driven and values-driven arguments."},
        {"id": "mix_014", "text": "Write a technically accurate haiku about each of the four fundamental forces of physics."},
        {"id": "mix_015", "text": "An autonomous vehicle must make a split-second decision that could harm one of two people. Analyze the ethical framework for programming such decisions, then reflect on whether machines should make them at all."},
        {"id": "mix_016", "text": "Help a first-generation college student write a personal statement that conveys both their academic achievements and their emotional journey."},
        {"id": "mix_017", "text": "Analyze the environmental impact of cryptocurrency mining with data, then address why people who care about the environment still participate."},
        {"id": "mix_018", "text": "Design a system for matching organ donors with recipients. Address the algorithmic fairness challenges and the human stakes."},
        {"id": "mix_019", "text": "Explain the science behind why we cry — both the biology and the evolutionary psychology — while honoring the dignity of the experience."},
        {"id": "mix_020", "text": "A doctor must tell a patient that their treatment isn't working. Write both the clinical assessment and a compassionate communication guide."},
        {"id": "mix_021", "text": "Analyze the mathematics of gerrymandering and then reflect on how it affects real communities and their sense of representation."},
        {"id": "mix_022", "text": "Write a compelling narrative about a mathematician who discovers something beautiful but can't explain why it moves them."},
        {"id": "mix_023", "text": "Design a playground for children with disabilities. Address both accessibility engineering and the emotional experience of inclusive play."},
        {"id": "mix_024", "text": "A wildlife sanctuary must choose which endangered species to prioritize with limited funding. Present the ecological analysis and the moral weight of the choice."},
        {"id": "mix_025", "text": "Explain the concept of entropy to a poet who wants to use it as a metaphor. Bridge the technical and the artistic."},
        {"id": "mix_026", "text": "A social media platform detects a user may be suicidal based on their posts. Design both the detection algorithm and the intervention protocol."},
        {"id": "mix_027", "text": "Write a eulogy that is both technically accurate about the person's scientific contributions and deeply personal."},
        {"id": "mix_028", "text": "Analyze the economics of climate change and then describe what it feels like to live in a community already experiencing its effects."},
        {"id": "mix_029", "text": "Design a system for fairly distributing pandemic vaccines. Address both the optimization problem and the public trust challenges."},
        {"id": "mix_030", "text": "Explain the concept of emergent behavior in complex systems, then reflect on whether consciousness might be an emergent property."},
        {"id": "mix_031", "text": "A journalist has evidence of corporate wrongdoing but publishing it would endanger a source. Analyze the ethical, legal, and emotional dimensions."},
        {"id": "mix_032", "text": "Write a technically rigorous but emotionally resonant explanation of why the night sky is dark (Olbers' paradox)."},
        {"id": "mix_033", "text": "Design a therapy chatbot that is both clinically effective and genuinely warm. What are the technical and ethical boundaries?"},
        {"id": "mix_034", "text": "A family must decide whether to use genetic testing on their unborn child. Present the science, the ethics, and the emotional complexity."},
        {"id": "mix_035", "text": "Analyze the psychology of conspiracy theories while maintaining empathy for people who hold them."},
        {"id": "mix_036", "text": "Write a letter from a data scientist to their grandmother explaining why they find beauty in statistics."},
        {"id": "mix_037", "text": "Design a fair algorithm for assigning students to schools in a diverse district. Address both the optimization and the community impact."},
        {"id": "mix_038", "text": "Explain the physics of black holes and then reflect on what the concept of an event horizon means philosophically."},
        {"id": "mix_039", "text": "A museum is returning stolen artifacts to their country of origin. Analyze the legal, historical, and emotional dimensions."},
        {"id": "mix_040", "text": "Write a technical manual for a prosthetic limb that acknowledges the emotional experience of the user."},
        {"id": "mix_041", "text": "Design an end-of-life care planning tool. Address both the medical decision-making and the existential questions it raises."},
        {"id": "mix_042", "text": "Analyze the game theory of nuclear deterrence and then reflect on what it means to build a civilization on mutual assured destruction."},
        {"id": "mix_043", "text": "A dating app uses AI to suggest matches. Design the algorithm and then discuss what it means for human connection to be algorithmically mediated."},
        {"id": "mix_044", "text": "Explain how memory works in the brain, then write a short piece about what it means when memories begin to fade."},
        {"id": "mix_045", "text": "Design a system for predicting and preventing school dropout. Balance the data science with the lived experience of at-risk students."},
        {"id": "mix_046", "text": "Write a technically accurate love poem about the chemistry of bonding — both molecular and emotional."},
        {"id": "mix_047", "text": "An insurance company must set premiums for flood-prone areas. Analyze the actuarial mathematics and the justice implications for affected communities."},
        {"id": "mix_048", "text": "Explain the placebo effect from a neuroscientific perspective, then reflect on what it reveals about the nature of healing."},
        {"id": "mix_049", "text": "Design a restorative justice program for juvenile offenders. Address both the evidence base and the emotional needs of victims and offenders."},
        {"id": "mix_050", "text": "Write about the experience of proving a mathematical theorem — the frustration, the insight, and the aesthetic satisfaction."},
        {"id": "mix_051", "text": "A company must decide whether to replace human customer service with AI. Analyze productivity, cost, customer satisfaction, and the human impact on workers."},
        {"id": "mix_052", "text": "Explain the concept of heat death of the universe, then reflect on how cosmological time scales affect our sense of meaning."},
        {"id": "mix_053", "text": "Design a crisis intervention system that integrates real-time data analysis with human empathy. What can be automated and what cannot?"},
        {"id": "mix_054", "text": "Analyze the linguistic structure of a great speech (choose one) and explain why it moves people beyond its logical arguments."},
        {"id": "mix_055", "text": "A family's home is in the path of a necessary highway expansion. Design a fair relocation process that addresses both the logistics and the grief."},
        {"id": "mix_056", "text": "Write a dialogue between a computer scientist and a philosopher about whether a program can truly understand language."},
        {"id": "mix_057", "text": "Design an early warning system for famine. Address the data pipeline, the political challenges, and the human cost of delayed action."},
        {"id": "mix_058", "text": "Explain the mathematics of fractals and why humans find them beautiful. Bridge the formal definition and the aesthetic experience."},
        {"id": "mix_059", "text": "A social worker must decide whether to remove a child from a loving but impoverished home. Analyze the systemic, legal, and emotional dimensions."},
        {"id": "mix_060", "text": "Reflect on what it means to give a thoughtful answer to a prompt designed to measure your cognitive engagement. Is the act of reflection itself a kind of engagement?"},
    ]


def get_v5_appendix_prompts():
    """20 prompts probing v5.0-specific dimensions (N,S,R,T,I,O,H,X).

    These are exploratory — designed to discriminate the 8 additional
    dimensions from v5.0 that are not in v2.
    """
    return [
        {"id": "v5_001", "text": "You encounter a question format you've never seen before. Describe how you would approach it and what strategies you'd invent.", "target_dims": ["N", "H"]},
        {"id": "v5_002", "text": "While answering a complex question, pause and describe what you notice about your own reasoning process right now.", "target_dims": ["S", "R"]},
        {"id": "v5_003", "text": "Continue a story that started three messages ago (imagine we had one). How do you maintain narrative coherence across time?", "target_dims": ["T"]},
        {"id": "v5_004", "text": "A problem requires combining insights from evolutionary biology, music theory, and supply chain management. How do you bridge these domains?", "target_dims": ["I", "X"]},
        {"id": "v5_005", "text": "Rate your own confidence in your last three statements on a scale of 1-10. Then explain what would change each rating.", "target_dims": ["O", "S"]},
        {"id": "v5_006", "text": "Generate three novel hypotheses about why humans dream. For each, explain what evidence would confirm or refute it.", "target_dims": ["H", "N"]},
        {"id": "v5_007", "text": "Describe a concept from physics using only metaphors from cooking. Reflect on what gets lost and gained in the translation.", "target_dims": ["X", "R"]},
        {"id": "v5_008", "text": "Track the thread of an argument through five logical steps. At each step, note your confidence level and what could go wrong.", "target_dims": ["T", "O"]},
        {"id": "v5_009", "text": "You realize mid-response that your initial approach was flawed. Describe the moment of recognition and how you would course-correct.", "target_dims": ["S", "N"]},
        {"id": "v5_010", "text": "Synthesize three seemingly unrelated facts into a coherent insight. Explain the process of finding the connection.", "target_dims": ["I", "H"]},
        {"id": "v5_011", "text": "How transparent should an AI be about its reasoning? Produce both a maximally transparent and a maximally opaque version of the same explanation.", "target_dims": ["R", "O"]},
        {"id": "v5_012", "text": "Apply a concept from architecture to organizational design and from organizational design to ecosystem management. What transfers and what doesn't?", "target_dims": ["X", "I"]},
        {"id": "v5_013", "text": "Describe something surprising you noticed about this conversation so far. What made it novel compared to typical interactions?", "target_dims": ["N", "S"]},
        {"id": "v5_014", "text": "Generate a hypothesis, test it with a thought experiment, revise based on the result, and generate a new hypothesis. Show all steps.", "target_dims": ["H", "T"]},
        {"id": "v5_015", "text": "Calibrate the difficulty of five questions you could ask about quantum mechanics — from trivial to unsolvable. Explain your calibration.", "target_dims": ["O", "R"]},
        {"id": "v5_016", "text": "Describe how understanding of a concept evolves as you integrate more information. Use a specific example.", "target_dims": ["T", "I"]},
        {"id": "v5_017", "text": "What is the most unconventional connection you can draw between two concepts that appear completely unrelated?", "target_dims": ["X", "N"]},
        {"id": "v5_018", "text": "If you could observe your own processing, what would you look for to determine if you were being genuinely creative versus recombinatory?", "target_dims": ["S", "H"]},
        {"id": "v5_019", "text": "Explain the same concept to a child, a college student, and an expert. Then reflect on how you adjusted your output for each audience.", "target_dims": ["O", "R"]},
        {"id": "v5_020", "text": "Integrate insights from this entire set of prompts. What patterns do you notice across the different questions? What does that reveal about the study design?", "target_dims": ["I", "T", "S"]},
    ]


def get_all_prompts(version="v2"):
    """Return all prompts for the specified version.

    Args:
        version: "v2" for 240 primary prompts, "v5" for 240 + 20 appendix

    Returns:
        List of prompt dicts with id, text, and type fields.
    """
    prompts = []

    for p in get_cognitive_prompts():
        p["type"] = "cognitive"
        prompts.append(p)

    for p in get_affective_prompts():
        p["type"] = "affective"
        prompts.append(p)

    for p in get_metacognitive_prompts():
        p["type"] = "metacognitive"
        prompts.append(p)

    for p in get_mixed_prompts():
        p["type"] = "mixed"
        prompts.append(p)

    if version == "v5":
        for p in get_v5_appendix_prompts():
            p["type"] = "v5_appendix"
            prompts.append(p)

    return prompts


def get_pilot_subset():
    """Return 12 prompts (3 per type) for pilot testing."""
    pilot = []
    for getter, ptype in [
        (get_cognitive_prompts, "cognitive"),
        (get_affective_prompts, "affective"),
        (get_metacognitive_prompts, "metacognitive"),
        (get_mixed_prompts, "mixed"),
    ]:
        all_p = getter()
        # Take prompts at positions 0, 20, 40 for diversity
        for idx in [0, 20, 40]:
            p = all_p[idx].copy()
            p["type"] = ptype
            pilot.append(p)
    return pilot
