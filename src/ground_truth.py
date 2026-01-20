GROUND_TRUTH_QUERIES = [
    # Artificial Intelligence document
    {
        "query": "What is knowledge representation and knowledge engineering in AI?",
        "expected_titles": ["Artificial intelligence"],
        "source_text": "Knowledge representation and knowledge engineering allow AI programs to answer questions intelligently and make deductions about real-world facts."
    },
    {
        "query": "How do AI systems perform reasoning and problem-solving?",
        "expected_titles": ["Artificial intelligence"],
        "source_text": "Early researchers developed algorithms that imitated step-by-step reasoning that humans use when they solve puzzles or make logical deductions."
    },
    
    # Machine Learning document
    {
        "query": "Who coined the term machine learning and when?",
        "expected_titles": ["Machine learning"],
        "source_text": "The term machine learning was coined in 1959 by Arthur Samuel, an IBM employee and pioneer in the field of computer gaming and artificial intelligence."
    },
    {
        "query": "What is the Cybertron learning machine developed by Raytheon?",
        "expected_titles": ["Machine learning"],
        "source_text": "By the early 1960s, an experimental learning machine with punched tape memory, called Cybertron, had been developed by Raytheon Company to analyse sonar signals, electrocardiograms, and speech patterns."
    },
    {
        "query": "What is empirical risk minimisation in machine learning?",
        "expected_titles": ["Machine learning", "Neural network (machine learning)"],
        "source_text": "Most traditional machine learning and deep learning algorithms can be described as empirical risk minimisation under this framework."
    },
    
    # Deep Learning document
    {
        "query": "Why is deep learning called deep? What does the word deep refer to?",
        "expected_titles": ["Deep learning"],
        "source_text": "The word deep in deep learning refers to the number of layers through which the data is transformed."
    },
    {
        "query": "What is credit assignment path (CAP) depth in neural networks?",
        "expected_titles": ["Deep learning"],
        "source_text": "More precisely, deep learning systems have a substantial credit assignment path (CAP) depth. The CAP is the chain of transformations from input to output."
    },
    {
        "query": "What are common deep learning network architectures?",
        "expected_titles": ["Deep learning"],
        "source_text": "Some common deep learning network architectures include fully connected networks, deep belief networks, recurrent neural networks, convolutional neural networks, generative adversarial networks, transformers, and neural radiance fields."
    },
    
    # Neural Network document
    {
        "query": "What is an artificial neuron and how does it work?",
        "expected_titles": ["Neural network (machine learning)"],
        "source_text": "A neural network consists of connected units or nodes called artificial neurons, which loosely model the neurons in the brain."
    },
    {
        "query": "What is the activation function in neural networks?",
        "expected_titles": ["Neural network (machine learning)"],
        "source_text": "The output of each neuron is computed by some non-linear function of the totality of its inputs, called the activation function."
    },
    {
        "query": "How many hidden layers does a deep neural network have?",
        "expected_titles": ["Neural network (machine learning)"],
        "source_text": "A network is typically called a deep neural network if it has at least two hidden layers."
    },
    
    # Natural Language Processing document  
    {
        "query": "What are the major tasks in NLP systems?",
        "expected_titles": ["Natural language processing"],
        "source_text": "Major processing tasks in an NLP system include: speech recognition, text classification, natural language understanding, and natural language generation."
    },
    {
        "query": "What was the Georgetown experiment in 1954?",
        "expected_titles": ["Natural language processing"],
        "source_text": "The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English."
    },
    {
        "query": "What is John Searle's Chinese room thought experiment about?",
        "expected_titles": ["Natural language processing"],
        "source_text": "The premise of symbolic NLP is often illustrated using John Searle's Chinese room thought experiment."
    },
    
    # Transformer document
    {
        "query": "What paper introduced the modern transformer architecture?",
        "expected_titles": ["Transformer (deep learning)"],
        "source_text": "The modern version of the transformer was proposed in the 2017 paper Attention Is All You Need by researchers at Google."
    },
    {
        "query": "What is multi-head attention mechanism in transformers?",
        "expected_titles": ["Transformer (deep learning)"],
        "source_text": "The transformer is an artificial neural network architecture based on the multi-head attention mechanism."
    },
    {
        "query": "What are GPT and BERT models?",
        "expected_titles": ["Transformer (deep learning)"],
        "source_text": "It has also led to the development of pre-trained systems, such as generative pre-trained transformers (GPTs) and BERT (bidirectional encoder representations from transformers)."
    },
    
    # Computer Vision document
    {
        "query": "What are subdisciplines of computer vision?",
        "expected_titles": ["Computer vision"],
        "source_text": "Subdisciplines of computer vision include scene reconstruction, object detection, event detection, activity recognition, video tracking, object recognition, 3D pose estimation."
    },
    {
        "query": "What types of image data does computer vision process?",
        "expected_titles": ["Computer vision"],
        "source_text": "Image data can take many forms, such as video sequences, views from multiple cameras, multi-dimensional data from a 3D scanner, 3D point clouds from LiDaR sensors."
    },
    
    # Reinforcement Learning document
    {
        "query": "What is the exploration-exploitation dilemma in reinforcement learning?",
        "expected_titles": ["Reinforcement learning"],
        "source_text": "The search for the optimal balance between these two strategies is known as the exploration–exploitation dilemma."
    },
    {
        "query": "What are the three basic machine learning paradigms?",
        "expected_titles": ["Reinforcement learning", "Machine learning"],
        "source_text": "Reinforcement learning is one of the three basic machine learning paradigms, alongside supervised learning and unsupervised learning."
    },
    {
        "query": "How is reinforcement learning environment typically modeled?",
        "expected_titles": ["Reinforcement learning"],
        "source_text": "The environment is typically stated in the form of a Markov decision process, as many reinforcement learning algorithms use dynamic programming techniques."
    },
]


def get_ground_truth_queries():
    """Return list of query strings."""
    return [item["query"] for item in GROUND_TRUTH_QUERIES]


def get_ground_truth_map():
    """Return mapping of query -> expected document titles."""
    return {
        item["query"]: item["expected_titles"] 
        for item in GROUND_TRUTH_QUERIES
    }
