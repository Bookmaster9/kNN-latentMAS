
def build_agent_message_sequential_latent_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    if role == "planner":
        user_prompt = f"""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: {question}

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:
"""
    
    elif role == "critic":
        user_prompt = f"""
Question: {question}

You are a Critic Agent to evaluate the correctness of the input plan for the given question and provide helpful feedback for improving the plan.
The plan information is provided in latent KV representation format. Review the plan and question and output:
(1) original plan contents
(2) constructive feedback on the original plan.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""
    
    elif role == "refiner":
        user_prompt = f"""
Question: {question}

You are a Refiner Agent to provide a refined step-by-step plan for solving the given question.
You are provided with:
(1) latent-format information: a previous plan with feedback
(2) text-format information: the input question you need to solve.

Based on the input, write a refined and improved plan to solve the question. Make sure your output plan is correct and concise.

Now, output your refined plan below:
"""
    
    elif role == "judger":
        user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve.

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""
        
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def build_agent_message_hierarchical_latent_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    if True:  # Simplified to only support GSM8K
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_sequential_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["text_mas"], "only for text_mas method"
    assert "qwen" in args.model_name.lower(), "only for qwen models"

    # truncate context if needed
    ctx = context[: args.text_mas_context_length]

    if role == "planner":
        user_content = f"""
You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

## Input Question:
{question}

Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answer.

## Format your response as follows:
Planner Agent's Output:
[Your detailed plan here]

Now output your plan to solve the question below:
"""

    elif role == "critic":
        user_content = f"""
You are a Critic Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan in text format.

Your job is to carefully evaluate the correctness and completeness of the plan and provide helpful feedback.

## Input Question:
{question}

## Plan from Planner Agent:
{ctx}

## Format your response as follows:
Critic Agent's Output:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""

    elif role == "refiner":
        user_content = f"""
You are a Refiner Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan together with Critic Agent's feedback in text format.

Your job is to incorporate the feedback and produce an improved, refined step-by-step plan.

## Input Question:
{question}

## Original Plan and Critic Feedback:
{ctx}

## Format your response as follows:
Refiner Agent's Output:
[Your refined and improved plan here]

Make sure your output plan is logically correct, concise, and sufficient to guide final problem solving.
Now, output your refined plan below:
"""

    elif role == "judger":
        user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_hierarchical_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["text_mas"], "this prompt only for text_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    if True:  # Simplified to only support GSM8K
        if role == "planner":
            user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

Input Question: {question}

Your response:
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_single_agent(question: str, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert args.method in ["baseline"], "this prompt only for baseline method (single agent)"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]

